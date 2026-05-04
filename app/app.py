# =============================================================================
# AI Data Model Designer — Main Application
# =============================================================================
# A Databricks App (Dash/Cytoscape) that analyzes Unity Catalog schemas
# and proposes normalized 3NF relational Entity-Relationship models using LLM.
# =============================================================================

import os
import re
import json
import time
import threading
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import dash
from dash import html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

from flask import request as flask_request
from databricks.sdk import WorkspaceClient

import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_TITLE = "AI Data Model Designer"

# SQL Warehouse — set by app.yaml env var
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

# LLM Endpoints — dynamically discovered from workspace (Claude models only)
LLM_FALLBACK_CHAIN = []
_llm_last_refresh = 0  # timestamp of last endpoint refresh
_LLM_REFRESH_INTERVAL = 300  # re-discover every 5 minutes

# Tuning parameters
PROFILE_WORKERS = 20
LLM_WORKERS = 20
LLM_GROUP_SIZE = 3
API_TIMEOUT = 300
MAX_TOKENS_SINGLE = 32000
MAX_TOKENS_GROUP = 16000
MAX_PAYLOAD_CHARS = 500000
POLL_INTERVAL_MS = 1200

# Color scheme
COLORS = {
    # 3NF entity types
    "core_entity": "#1e3a8a",   # Dark Navy
    "weak_entity": "#0ea5e9",   # Sky Blue
    "associative": "#f97316",   # Orange
    "reference": "#64748b",     # Slate
    # Dimensional entity types
    "fact": "#2563eb",          # Blue
    "dimension": "#16a34a",     # Green
    "bridge": "#ea580c",        # Deep Orange
    "aggregate": "#7c3aed",     # Purple
    # Column roles
    "pk": "#eab308",            # Gold
    "fk": "#2563eb",            # Blue
    "measure": "#16a34a",       # Green
    "attribute": "#94a3b8",     # Gray
}

# Thread-local storage for auth tokens
_tls = threading.local()

# Global job state for progress tracking
_job = {
    "active": False,
    "steps": [],
    "result": None,
    "error": None,
    "cancelled": False,
    "start_time": None,
}

# =============================================================================
# DYNAMIC LLM ENDPOINT DISCOVERY (Claude-only, auto-refresh)
# =============================================================================

def _discover_endpoints():
    """Auto-discover ALL Claude serving endpoints in the workspace.

    Finds every endpoint matching 'claude' (case-insensitive), sorts by:
      1. Tier: opus > sonnet > haiku > other
      2. Version: higher numbers first (e.g. 4-7 > 4-6 > 4-5)
      3. Name (alphabetical tiebreak)

    No hardcoded model names — always uses what's live in the workspace.
    """
    global LLM_FALLBACK_CHAIN, _llm_last_refresh
    try:
        w = WorkspaceClient()
        all_eps = list(w.serving_endpoints.list())

        # Match ANY endpoint with 'claude' in its name
        candidates = []
        for ep in all_eps:
            name = ep.name or ""
            if re.search(r"claude", name, re.IGNORECASE):
                candidates.append(name)

        if not candidates:
            print("[app.py] WARNING: No Claude endpoints found in workspace.")
            print("[app.py] The app will not be able to generate models until Claude endpoints are available.")
            LLM_FALLBACK_CHAIN = []
            _llm_last_refresh = time.time()
            return

        def _sort_key(name):
            nl = name.lower()
            # Tier priority: opus=0, sonnet=1, haiku=2, other=3
            if "opus" in nl:
                tier = 0
            elif "sonnet" in nl:
                tier = 1
            elif "haiku" in nl:
                tier = 2
            else:
                tier = 3
            # Extract version numbers (e.g. '4-7' → [4,7], '3-5' → [3,5])
            nums = re.findall(r"(\d+)", nl.split("claude")[-1])
            # Pad to 3 so sonnet-4 → (-4,0,0) sorts AFTER sonnet-4-6 → (-4,-6,0)
            padded = [-int(n) for n in nums] if nums else [0]
            while len(padded) < 3:
                padded.append(0)
            version = tuple(padded)
            return (tier, version, name)

        candidates.sort(key=_sort_key)
        LLM_FALLBACK_CHAIN = candidates  # Use ALL discovered Claude endpoints
        _llm_last_refresh = time.time()
        print(f"[app.py] Discovered {len(candidates)} Claude endpoint(s): {candidates}")
    except Exception as e:
        print(f"[app.py] Warning: Could not discover endpoints: {e}")
        # Do NOT set a hardcoded fallback — keep whatever was last discovered
        if not LLM_FALLBACK_CHAIN:
            print("[app.py] No endpoints available. Generate will fail until endpoints are discoverable.")


def _refresh_endpoints_if_stale():
    """Re-discover endpoints if the cache is older than _LLM_REFRESH_INTERVAL seconds."""
    if time.time() - _llm_last_refresh > _LLM_REFRESH_INTERVAL:
        _discover_endpoints()


def _get_model_pools():
    """Return tiered model pools: Sonnet for Map, best Opus for Reduce.

    Map pool:  Only the LATEST Sonnet endpoint(s) — older versions go to fallback.
    Reduce pool: Single best Opus endpoint (no racing — saves cost).

    Fallback: Map falls back to older Sonnet → Opus → Haiku.
              Reduce falls back to Sonnet → Haiku.

    LLM_FALLBACK_CHAIN is pre-sorted: within each tier, latest version first.
    """
    _refresh_endpoints_if_stale()

    opus_models = [m for m in LLM_FALLBACK_CHAIN if "opus" in m.lower()]
    sonnet_models = [m for m in LLM_FALLBACK_CHAIN if "sonnet" in m.lower()]
    haiku_models = [m for m in LLM_FALLBACK_CHAIN if "haiku" in m.lower()]

    # --- Map pool: ONLY latest-version Sonnet ---
    # sonnet_models is already sorted latest-first by _discover_endpoints().
    # Extract the version of the first (latest) Sonnet, then take all with that version.
    # e.g. [sonnet-4-5] is latest → only use sonnet-4-5, not sonnet-4-1 or sonnet-3-7.
    map_latest = []
    if sonnet_models:
        _latest = sonnet_models[0]
        # Extract version signature: e.g. "sonnet-4-5" → ("4","5")
        _ver = re.findall(r"(\d+)", _latest.lower().split("sonnet")[-1])
        while len(_ver) < 3:
            _ver.append("0")
        _latest_ver = tuple(_ver)
        for m in sonnet_models:
            m_ver_raw = re.findall(r"(\d+)", m.lower().split("sonnet")[-1])
            while len(m_ver_raw) < 3:
                m_ver_raw.append("0")
            m_ver = tuple(m_ver_raw)
            if m_ver == _latest_ver:
                map_latest.append(m)
    map_primary = map_latest if map_latest else opus_models[:3]
    # Fallback: older Sonnet → Opus → Haiku
    older_sonnet = [m for m in sonnet_models if m not in map_primary]
    map_fallback = older_sonnet + [m for m in opus_models + haiku_models if m not in map_primary]

    # --- Reduce pool: best single Opus (no racing — saves cost) ---
    reduce_primary = opus_models[:1] if opus_models else sonnet_models[:1]
    reduce_fallback = [m for m in sonnet_models + haiku_models if m not in reduce_primary]

    return {
        "map": {"primary": map_primary, "fallback": map_fallback},
        "reduce": {"primary": reduce_primary, "fallback": reduce_fallback},
    }


try:
    _discover_endpoints()
except Exception as _e:
    print(f"[app.py] WARNING: Endpoint discovery failed at startup: {_e}")

# =============================================================================
# AUTHENTICATION & SQL EXECUTION
# =============================================================================

def _get_token(explicit_token=None):
    """Get OAuth token with priority: explicit > thread-local > flask request > SP."""
    if explicit_token:
        return explicit_token
    token = getattr(_tls, "token", None)
    if token:
        return token
    try:
        token = flask_request.headers.get("x-forwarded-access-token")
        if token:
            return token
    except RuntimeError:
        pass  # Outside request context — expected in background threads
    # Last resort: service principal token
    try:
        sp = WorkspaceClient()
        sp_headers = sp.config.authenticate()
        auth_val = sp_headers.get("Authorization", "")
        if auth_val.startswith("Bearer "):
            return auth_val[7:]
    except Exception:
        pass
    return None


def run_sql(stmt, params=None, token=None):
    """Execute SQL via SDK Statement Execution API as the logged-in user.

    Uses WorkspaceClient(host, token, auth_type='pat') to avoid auth
    conflicts in the Databricks App runtime.
    Returns list of dicts.
    """
    tok = _get_token(token)
    if not tok:
        raise RuntimeError("No authentication token available")
    if not WAREHOUSE_ID:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID not set")

    _host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if _host and not _host.startswith("http"):
        _host = f"https://{_host}"

    # Create a per-request WorkspaceClient with the user's OAuth token
    wc = WorkspaceClient(host=_host, token=tok, auth_type="pat")
    resp = wc.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=stmt,
        wait_timeout="50s",
    )

    # Poll if still running (warehouse may be cold-starting)
    state = resp.status.state.value
    if state in ("PENDING", "RUNNING"):
        sid = resp.statement_id
        for _ in range(60):
            time.sleep(5)
            resp = wc.statement_execution.get_statement(sid)
            state = resp.status.state.value
            if state not in ("PENDING", "RUNNING"):
                break

    if state != "SUCCEEDED":
        err_msg = getattr(resp.status, "error", None)
        raise RuntimeError(f"SQL failed ({state}): {err_msg}")

    cols = [c.name for c in resp.manifest.schema.columns]
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return [dict(zip(cols, row)) for row in data]


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def _call_chat_model(prompt, token, max_tokens=MAX_TOKENS_SINGLE, model_chain=None):
    """Call a Foundation Model endpoint with fallback chain."""
    if model_chain is None:
        _refresh_endpoints_if_stale()
    chain = model_chain or LLM_FALLBACK_CHAIN
    if not chain:
        raise RuntimeError("No LLM endpoints configured")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    last_error = None
    _t0 = time.time()
    _short_model = lambda m: m.replace("databricks-claude-", "")

    for model in chain:
        url = f"{host}/serving-endpoints/{model}/invocations"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        is_opus = "opus" in model.lower()
        max_attempts = 3 if is_opus else 2  # Opus gets extra retry
        print(f"[LLM] → {_short_model(model)} | prompt={len(prompt):,} chars | max_tokens={max_tokens} | attempts={max_attempts}", flush=True)

        for attempt in range(max_attempts):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    content_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    elapsed = time.time() - _t0
                    print(f"[LLM] ← {_short_model(model)} | {len(content_text):,} chars | {elapsed:.1f}s", flush=True)
                    return content_text
                elif resp.status_code == 429:
                    backoff = min(15 * (2 ** attempt), 60)  # 15s, 30s, 60s
                    print(f"[LLM] ⚠ {_short_model(model)} → 429 rate limited, backoff {backoff}s (attempt {attempt+1}/{max_attempts})", flush=True)
                    if attempt < max_attempts - 1:
                        time.sleep(backoff)
                        continue
                    last_error = f"429 rate limited on {model} after {max_attempts} attempts"
                    break
                elif resp.status_code in (500, 502, 503):
                    print(f"[LLM] ⚠ {_short_model(model)} → {resp.status_code} service error, trying next...", flush=True)
                    last_error = f"{resp.status_code} on {model}"
                    break  # Fall to next model
                elif resp.status_code == 400:
                    # Check if it's a payload size issue (context length exceeded)
                    err_text = resp.text[:500].lower()
                    prompt_large = len(prompt) > 100_000  # >100K chars likely exceeds context
                    size_keywords = ("too long", "context length", "token", "maximum", "too large",
                                     "max_tokens", "content_length", "length exceeded", "input too")
                    if prompt_large or any(k in err_text for k in size_keywords):
                        raise OverflowError(f"Payload too large for {model} ({len(prompt):,} chars): {resp.text[:150]}")
                    raise RuntimeError(f"400 on {model}: {resp.text[:200]}")
                elif resp.status_code in (401, 403):
                    raise RuntimeError(f"{resp.status_code} on {model}: {resp.text[:200]}")
                else:
                    print(f"[LLM] ✗ {_short_model(model)} → {resp.status_code}", flush=True)
                    last_error = f"{resp.status_code} on {model}"
                    break
            except requests.Timeout:
                print(f"[LLM] ✗ {_short_model(model)} → timeout", flush=True)
                last_error = f"Timeout on {model}"
                break
            except RuntimeError:
                raise
            except Exception as e:
                print(f"[LLM] ✗ {_short_model(model)} → {str(e)[:80]}", flush=True)
                last_error = f"Error on {model}: {str(e)[:100]}"
                break

    raise RuntimeError(f"All LLM endpoints failed. Last error: {last_error}")


def _call_chat_model_race(prompt, token, max_tokens=MAX_TOKENS_SINGLE, use_models=None):
    """Fire prompt to selected endpoints simultaneously, return first success.

    Uses only top Opus models by default (not sonnet/haiku).
    Non-blocking: returns immediately when a winner is found — does NOT
    wait for remaining HTTP requests to finish.
    """
    _refresh_endpoints_if_stale()
    if not LLM_FALLBACK_CHAIN:
        raise RuntimeError("No LLM endpoints configured")

    # Select models: explicit list, or top 3 Opus by default (for Reduce phase)
    if use_models:
        race_models = use_models
    else:
        pools = _get_model_pools()
        race_models = pools["reduce"]["primary"]  # Top 3 Opus for racing

    _short = lambda m: m.replace("databricks-claude-", "")
    ep_names = [_short(m) for m in race_models]
    print(f"[RACE] 🏁 Racing {len(race_models)} endpoints: {', '.join(ep_names)} | prompt={len(prompt):,} chars", flush=True)
    _race_t0 = time.time()

    # Background timer — updates modal every 10s so user sees progress
    _race_done = threading.Event()
    def _timer_update():
        while not _race_done.is_set():
            _race_done.wait(10)
            if not _race_done.is_set():
                elapsed = time.time() - _race_t0
                _update_step(f"⏳ Waiting for LLM response... {elapsed:.0f}s elapsed ({', '.join(ep_names)} racing)")
    _timer_thread = threading.Thread(target=_timer_update, daemon=True)
    _timer_thread.start()

    errors = {}
    _responded = 0
    _winner_result = [None]  # Use list for thread-safe mutable

    def _call_one(model):
        return _call_chat_model(prompt, token, max_tokens, model_chain=[model])

    # NON-BLOCKING: create pool without context manager so we don't wait for stragglers
    pool = ThreadPoolExecutor(max_workers=len(race_models))
    try:
        futures = {pool.submit(_call_one, m): m for m in race_models}
        for future in as_completed(futures):
            model = futures[future]
            _responded += 1
            try:
                result = future.result()
                if result:
                    elapsed = time.time() - _race_t0
                    print(f"[RACE] 🏆 Winner: {_short(model)} | {len(result):,} chars | {elapsed:.1f}s", flush=True)
                    _race_done.set()
                    _update_step(f"🏆 {_short(model)} responded first — {len(result):,} chars in {elapsed:.0f}s", status="done")
                    _winner_result[0] = result
                    # Don't wait for others — shutdown without blocking
                    pool.shutdown(wait=False, cancel_futures=True)
                    return result
            except Exception as e:
                elapsed = time.time() - _race_t0
                err_short = str(e)[:60]
                print(f"[RACE] ✗ {_short(model)} failed ({elapsed:.1f}s): {err_short}", flush=True)
                errors[model] = str(e)
                remaining = len(futures) - _responded
                if remaining > 0:
                    _update_step(f"⚠ {_short(model)} failed — waiting for {remaining} more endpoint(s)... ({elapsed:.0f}s)")
                else:
                    _update_step(f"⚠ All {len(race_models)} endpoints failed ({elapsed:.0f}s)", status="error")
    finally:
        _race_done.set()
        pool.shutdown(wait=False, cancel_futures=True)

    raise RuntimeError(f"Race failed on all {len(race_models)} endpoints: {errors}")


# =============================================================================
# PROFILING ENGINE
# =============================================================================

def _get_metadata(catalog, schema, token):
    """Get table/column metadata from information_schema."""
    sql = f"""
    SELECT table_name, column_name, data_type, is_nullable
    FROM `{catalog}`.information_schema.columns
    WHERE table_schema = '{schema}'
    ORDER BY table_name, ordinal_position
    """
    rows = run_sql(sql, token=token)
    tables = defaultdict(list)
    for r in rows:
        tables[r["table_name"]].append({
            "name": r["column_name"],
            "type": r["data_type"],
            "nullable": r["is_nullable"] == "YES",
        })
    return dict(tables)


def _get_row_count(catalog, schema, table, token):
    """Get row count via DESCRIBE DETAIL (Delta) or estimate."""
    try:
        rows = run_sql(f"DESCRIBE DETAIL `{catalog}`.`{schema}`.`{table}`", token=token)
        if rows:
            return int(rows[0].get("numRecords", 0))
    except Exception:
        pass
    return None  # Caller will estimate


def _profile_single_table(catalog, schema, tname, cols, token=None):
    """Profile a single table using TABLESAMPLE."""
    _tls.token = token  # Set for this worker thread

    fqn = f"`{catalog}`.`{schema}`.`{tname}`"
    select_parts = ["COUNT(*) AS __row_count__"]

    date_types = {"DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ"}
    string_types = {"STRING", "VARCHAR", "CHAR"}
    string_cols = [c for c in cols if c["type"].upper() in string_types][:4]  # Top 4

    for c in cols:
        cn = c["name"]
        select_parts.append(f"COUNT(DISTINCT `{cn}`) AS `dist__{cn}`")
        select_parts.append(f"SUM(CASE WHEN `{cn}` IS NULL THEN 1 ELSE 0 END) AS `nulls__{cn}`")
        if c["type"].upper() in date_types:
            select_parts.append(f"CAST(MIN(`{cn}`) AS STRING) AS `min__{cn}`")
            select_parts.append(f"CAST(MAX(`{cn}`) AS STRING) AS `max__{cn}`")

    # Sample values for string columns
    for c in string_cols:
        cn = c["name"]
        select_parts.append(
            f"(SELECT CONCAT_WS('|||', COLLECT_LIST(val)) "
            f"FROM (SELECT DISTINCT CAST(`{cn}` AS STRING) AS val "
            f"FROM {fqn} WHERE `{cn}` IS NOT NULL LIMIT 3)) AS `sample__{cn}`"
        )

    sql = f"SELECT {', '.join(select_parts)} FROM {fqn} TABLESAMPLE (10 ROWS)"

    try:
        rows = run_sql(sql, token=token)
        if not rows:
            return {"table": tname, "error": "No rows returned", "cols": cols}

        r = rows[0]
        sampled_count = int(r.get("__row_count__", 0) or 0)

        # Use sampled count directly (from TABLESAMPLE 10 ROWS — no extra SQL call)
        actual_count = sampled_count

        col_profiles = []
        for c in cols:
            cn = c["name"]
            total = max(sampled_count, 1)
            distinct = int(r.get(f"dist__{cn}", 0) or 0)
            nulls = int(r.get(f"nulls__{cn}", 0) or 0)
            profile = {
                "name": cn,
                "type": c["type"],
                "null_pct": round(nulls / total * 100, 1),
                "cardinality_ratio": round(distinct / total, 3) if total > 0 else 0,
            }
            # Date ranges
            min_val = r.get(f"min__{cn}")
            max_val = r.get(f"max__{cn}")
            if min_val:
                profile["min"] = str(min_val)
            if max_val:
                profile["max"] = str(max_val)
            # Sample values
            sample = r.get(f"sample__{cn}")
            if sample:
                profile["samples"] = [s for s in str(sample).split("|||") if s][:3]
            col_profiles.append(profile)

        return {
            "table": tname,
            "row_count": actual_count,
            "columns": col_profiles,
        }

    except Exception as e:
        return {"table": tname, "error": str(e)[:200], "cols": cols}


def _profile_schema(catalog, schema, token):
    """Profile all tables in a schema. Returns profiling payload."""
    _add_step(f"Connecting to `{catalog}`.`{schema}`...")
    metadata = _get_metadata(catalog, schema, token)

    if not metadata:
        _add_step("No tables found in schema", status="error")
        return None

    _update_step(f"Connected — found {len(metadata)} tables in `{catalog}`.`{schema}`", status="done")
    _add_step(f"Profiling tables in parallel (up to {PROFILE_WORKERS} concurrent)...")

    profiles = []
    completed = 0
    total_rows = 0
    total_cols = 0

    with ThreadPoolExecutor(max_workers=PROFILE_WORKERS) as pool:
        futures = {
            pool.submit(_profile_single_table, catalog, schema, tname, cols, token): (tname, cols)
            for tname, cols in metadata.items()
        }
        for future in as_completed(futures):
            tname, cols = futures[future]
            try:
                result = future.result()
                profiles.append(result)
                rc = result.get("row_count", 0)
                cc = len(result.get("columns", cols))
                total_rows += rc if isinstance(rc, int) else 0
                total_cols += cc
                rc_str = f"~{rc:,} rows" if isinstance(rc, int) else "? rows"
                _add_step(f"Profiled {tname} ({cc} cols, {rc_str})", status="done")
            except Exception as e:
                profiles.append({"table": tname, "error": str(e)[:200]})
                _add_step(f"Failed to profile {tname}: {str(e)[:60]}", status="error")
            completed += 1

    _add_step(f"Profiled {len(profiles)} tables ({total_cols} columns, {total_rows:,} total rows)", status="done")
    return profiles, catalog, schema



def _build_per_table_payloads(profiles, catalog, schema):
    """Build individual per-table payloads directly from profile dicts.

    Bypasses the monolithic string concatenation + truncation bottleneck.
    Each table gets its own payload with the schema header prepended.
    This ensures ALL profiled tables reach the LLM, regardless of total size.
    """
    header = f"Schema: `{catalog}`.`{schema}`\n"
    per_table = {}
    for p in profiles:
        tname = p.get("table", "unknown")
        if p.get("error"):
            continue  # Skip tables that failed profiling
        lines = [header]
        rc = p.get("row_count", "?")
        lines.append(f"\nTable: {tname} (~{rc:,} rows)" if isinstance(rc, int) else f"\nTable: {tname}")
        for c in p.get("columns", []):
            parts = [f"  - {c['name']} ({c['type']}): null={c.get('null_pct', 0)}%, card_ratio={c.get('cardinality_ratio', 0)}"]
            if c.get("min"):
                parts.append(f" range=[{c['min']} .. {c.get('max', '?')}]")
            if c.get("samples"):
                parts.append(f" samples={c['samples']}")
            lines.append("".join(parts))
        per_table[tname] = "\n".join(lines)
    return per_table

def _build_profiling_payload(profiles, catalog, schema):
    """Build the text payload sent to LLM."""
    lines = [f"Schema: `{catalog}`.`{schema}`\n"]
    for p in profiles:
        tname = p.get("table", "unknown")
        if p.get("error"):
            lines.append(f"\nTable: {tname} (ERROR: {p['error']})")
            continue
        rc = p.get("row_count", "?")
        lines.append(f"\nTable: {tname} (~{rc:,} rows)" if isinstance(rc, int) else f"\nTable: {tname}")
        for c in p.get("columns", []):
            parts = [f"  - {c['name']} ({c['type']}): null={c.get('null_pct', 0)}%, card_ratio={c.get('cardinality_ratio', 0)}"]
            if c.get("min"):
                parts.append(f" range=[{c['min']} .. {c.get('max', '?')}]")
            if c.get("samples"):
                parts.append(f" samples={c['samples']}")
            lines.append("".join(parts))

    payload = "\n".join(lines)
    if len(payload) > MAX_PAYLOAD_CHARS:
        payload = payload[:MAX_PAYLOAD_CHARS] + "\n... (truncated)"
    return payload


# =============================================================================
# LLM PROMPTS
# =============================================================================

def _build_system_prompt(model_type="3nf"):
    if model_type == "dimensional":
        return (
            "You are a senior enterprise data architect with 20 years of experience in dimensional modeling "
            "using the Kimball methodology. You specialize in star schemas, snowflake schemas, fact tables, "
            "dimension tables, bridge tables, and aggregate tables. "
            "Your job is to RECOMMEND a best-practice dimensional model — not merely echo existing table names. "
            "Rename tables and columns to follow professional Kimball naming conventions. "
            "You are precise, thorough, and always produce valid JSON."
        )
    return (
        "You are a senior enterprise data architect with 20 years of experience in relational database design, "
        "3rd Normal Form normalization, and Entity-Relationship modeling. "
        "Your job is to RECOMMEND a best-practice data model — not merely echo existing table names. "
        "Rename tables and columns to follow professional naming conventions. "
        "You are precise, thorough, and always produce valid JSON."
    )


def _build_analysis_prompt(payload, catalog, schema, model_type="3nf"):
    sys = _build_system_prompt(model_type)
    if model_type == "dimensional":
        rules = """RULES:
1. Design using Kimball dimensional modeling: identify facts (measurable business events) and dimensions (descriptive context).
2. Classify each table as exactly ONE of: fact, dimension, bridge, aggregate.
   - fact: Central event/transaction tables with measurable metrics (e.g. sales_fact, order_fact)
   - dimension: Descriptive context tables (e.g. dim_customer, dim_product, dim_date)
   - bridge: Tables resolving M:N relationships between facts and dimensions (e.g. bridge_customer_group)
   - aggregate: Pre-aggregated summary tables for performance (e.g. agg_monthly_sales)
3. ALWAYS include a dim_date dimension if date/time columns exist.
4. Column roles: pk (primary/surrogate key), measure (numeric additive/semi-additive metrics), attribute (descriptive fields).
5. Use naming convention: fact_ prefix for facts, dim_ prefix for dimensions, bridge_ for bridges, agg_ for aggregates.
6. Do NOT include FK relationships — they will be determined globally in a later step.
7. Be CONCISE in descriptions (15 words max each).
8. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP."""
        json_tmpl = """{
  "proposed_tables": [
    {
      "table_name": "entity_name",
      "table_type": "fact|dimension|bridge|aggregate",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|measure|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }
      ]
    }
  ],
  "columns_dropped": [{"source_table": "t", "column": "c", "reason": "why"}],
  "data_summary": "3-5 sentences written for a BUSINESS ANALYST audience (not engineers). Describe: what business domain this data represents, what key business processes or activities are tracked, what business questions this data can answer, and how the entities relate in business terms. Use plain language — say 'tracks customer orders and delivery status' not 'normalized transactional entity with surrogate keys'. Avoid technical jargon like canonical, discriminator, denormalized, surrogate, natural key, projection."
}"""
        intro = f"Analyze the following schema from `{catalog}`.`{schema}` and design a dimensional model (Kimball star schema)."
    else:
        rules = """RULES:
1. Normalize to strict 3rd Normal Form. Eliminate all transitive dependencies.
2. Classify each entity as exactly ONE of: core_entity, weak_entity, associative, reference.
   - core_entity: Independent entities with strong PKs (e.g. customer, product, order)
   - weak_entity: Entities that depend on a parent for identity (e.g. order_line, address)
   - associative: Junction tables resolving M:N relationships (e.g. student_course)
   - reference: Lookup/code tables with low cardinality, rarely changing (e.g. order_status, country)
3. NAMING CONVENTIONS — this is a RECOMMENDED model, not a mirror of existing tables:
   a. Use SINGULAR nouns for entity names (order, not orders; restaurant, not restaurants; payment, not payments).
   b. Use snake_case (lowercase with underscores).
   c. PK column: <entity_name>_id (e.g. order_id for the order entity).
   d. Reference/lookup tables: use _type, _status, or _category suffix (e.g. payment_method_type, order_status).
   e. Keep names descriptive — no abbreviations (delivery_driver, not dd).
   f. FK columns should match the PK name of the referenced entity (e.g. order.restaurant_id references restaurant.restaurant_id).
4. Column roles must be ONLY: pk or attribute. Do NOT extract FK relationships in this step.
5. Be CONCISE in descriptions (15 words max each).
6. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP.
7. Do NOT include a "relationships" array — relationships will be determined globally in a later step."""
        json_tmpl = """{
  "proposed_tables": [
    {
      "table_name": "entity_name",
      "table_type": "core_entity|weak_entity|associative|reference",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }
      ]
    }
  ],
  "columns_dropped": [{"source_table": "t", "column": "c", "reason": "why"}],
  "data_summary": "3-5 sentences written for a BUSINESS ANALYST audience (not engineers). Describe: what business domain this data represents, what key business processes or activities are tracked, what business questions this data can answer, and how the entities relate in business terms. Use plain language — say 'tracks customer orders and delivery status' not 'normalized transactional entity with surrogate keys'. Avoid technical jargon like canonical, discriminator, denormalized, surrogate, natural key, projection."
}"""
        intro = f"Analyze the following schema from `{catalog}`.`{schema}` and design a strictly normalized 3NF relational model."

    return f"""{sys}

{intro}

{rules}

PROFILING DATA:
{payload}

Return ONLY valid JSON with this exact structure:
{json_tmpl}

Return ONLY the JSON. No markdown fences. No extra text."""

def _build_group_prompt(payload, catalog, schema, all_table_names, model_type="3nf"):
    all_names = ", ".join(all_table_names)
    sys = _build_system_prompt(model_type)
    if model_type == "dimensional":
        rules = """RULES:
1. Design using Kimball dimensional modeling: facts (measurable events) and dimensions (descriptive context).
2. Classify each table as exactly ONE of: fact, dimension, bridge, aggregate.
   - fact: Central event/transaction tables with measurable metrics (e.g. sales_fact)
   - dimension: Descriptive context tables (e.g. dim_customer, dim_product, dim_date)
   - bridge: Tables resolving M:N relationships between facts and dimensions
   - aggregate: Pre-aggregated summary tables for performance
3. ALWAYS include a dim_date dimension if any date/time columns exist in the schema.
4. Column roles: pk (primary/surrogate key), measure (numeric additive/semi-additive metrics), attribute (descriptive).
5. Use naming: fact_ prefix for facts, dim_ for dimensions, bridge_ for bridges, agg_ for aggregates.
6. Do NOT include FK relationships — they will be determined globally later.
7. Be CONCISE in descriptions (15 words max).
8. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP."""
        json_tmpl = """{
  "proposed_tables": [
    {
      "table_name": "entity_name",
      "table_type": "fact|dimension|bridge|aggregate",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|measure|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }
      ]
    }
  ],
  "columns_dropped": [{"source_table": "t", "column": "c", "reason": "why"}],
  "data_summary": "3-5 sentences written for a BUSINESS ANALYST audience (not engineers). Describe: what business domain this data represents, what key business processes or activities are tracked, what business questions this data can answer, and how the entities relate in business terms. Use plain language — say 'tracks customer orders and delivery status' not 'normalized transactional entity with surrogate keys'. Avoid technical jargon like canonical, discriminator, denormalized, surrogate, natural key, projection."
}"""
        intro = f"Analyze the following tables and propose a dimensional model (Kimball star schema)."
    else:
        rules = """RULES:
1. Normalize to strict 3rd Normal Form. Eliminate all transitive dependencies.
2. Classify each entity as exactly ONE of: core_entity, weak_entity, associative, reference.
   - core_entity: Independent entities with strong PKs (e.g. customer, product, order)
   - weak_entity: Entities that depend on a parent for identity (e.g. order_line, address)
   - associative: Junction tables resolving M:N relationships (e.g. student_course)
   - reference: Lookup/code tables with low cardinality, rarely changing (e.g. order_status, country)
3. NAMING CONVENTIONS — this is a RECOMMENDED model, not a mirror of existing tables:
   a. Use SINGULAR nouns for entity names (order, not orders; restaurant, not restaurants; payment, not payments).
   b. Use snake_case (lowercase with underscores).
   c. PK column: <entity_name>_id (e.g. order_id for the order entity).
   d. Reference/lookup tables: use _type, _status, or _category suffix (e.g. payment_method_type, order_status).
   e. Keep names descriptive — no abbreviations (delivery_driver, not dd).
   f. FK columns should match the PK name of the referenced entity (e.g. order.restaurant_id references restaurant.restaurant_id).
4. Column roles must be ONLY: pk or attribute. Do NOT extract FK relationships in this step.
5. Be CONCISE in descriptions (15 words max each).
6. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP.
7. Do NOT include a "relationships" array — relationships will be determined globally in a later step."""
        json_tmpl = """{
  "proposed_tables": [
    {
      "table_name": "entity_name",
      "table_type": "core_entity|weak_entity|associative|reference",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }
      ]
    }
  ],
  "columns_dropped": [{"source_table": "t", "column": "c", "reason": "why"}],
  "data_summary": "3-5 sentences written for a BUSINESS ANALYST audience (not engineers). Describe: what business domain this data represents, what key business processes or activities are tracked, what business questions this data can answer, and how the entities relate in business terms. Use plain language — say 'tracks customer orders and delivery status' not 'normalized transactional entity with surrogate keys'. Avoid technical jargon like canonical, discriminator, denormalized, surrogate, natural key, projection."
}"""
        intro = "Analyze the following tables and propose normalized 3NF entity definitions."

    return f"""{sys}

You are analyzing a SUBSET of tables from `{catalog}`.`{schema}`.
All tables in the schema: {all_names}

{intro}
Note cross-references to tables NOT in this subset — they will be merged later.

{rules}

PROFILING DATA:
{payload}

Return ONLY valid JSON with this EXACT structure (use these EXACT key names):
{json_tmpl}
"""

def _build_merge_prompt(group_results, catalog, schema):
    """DEPRECATED: LLM-based merge is no longer used. Kept as stub for compatibility."""
    raise NotImplementedError("LLM merge replaced by programmatic merge + Reduce phase")




# =============================================================================
# REDUCE PHASE — Global Relationship Mapping (Opus)
# =============================================================================

def _build_schema_catalog(proposed_tables):
    """Build a compressed schema catalog from Map phase output.

    Structure-only: table name, type, columns with roles and types.
    No sample values, no null%, no cardinality — just what Opus needs
    to infer FK→PK relationships.
    ~400 chars per table → 200 tables ≈ 80K chars ≈ 20K tokens.
    """
    lines = []
    for t in proposed_tables:
        tname = t.get("table_name", "unknown")
        ttype = t.get("table_type", "core_entity")
        cols = t.get("columns", [])
        pk_cols = [c["name"] for c in cols if c.get("role", "").lower() == "pk"]
        attr_cols = [f'{c["name"]} ({c.get("data_type", "STRING")})' for c in cols if c.get("role", "").lower() != "pk"]

        lines.append(f"Entity: {tname} [{ttype}]")
        if pk_cols:
            lines.append(f"  PK: {', '.join(pk_cols)}")
        if attr_cols:
            lines.append(f"  Attributes: {', '.join(attr_cols)}")
    return "\n".join(lines)


def _build_reduce_prompt(compressed_catalog, catalog, schema, model_type="3nf"):
    """Build the Reduce phase prompt for global relationship mapping."""
    if model_type == "dimensional":
        context = f"""You are an Enterprise Data Architect specializing in Kimball dimensional modeling.

I am providing a complete dimensional model schema catalog for `{catalog}`.`{schema}`.
Each table lists its columns and Primary Keys.

Your ONLY task: identify EVERY foreign key relationship (fact→dimension joins, bridge links).

RULES:
1. Every fact table FK MUST reference a dimension PK (star schema joins).
2. Bridge tables should have at least 2 FK relationships (one to fact/dim, one to dim).
3. Do NOT invent columns — only use columns that exist in the catalog.
4. Include cardinality (1:1, 1:N, M:N).
5. Use action-verb descriptions (e.g., "Each sale references one product dimension").
6. Ensure no cross-table relationship is missed."""
    else:
        context = f"""You are an Enterprise Data Architect specializing in relational integrity.

I am providing a complete normalized 3NF schema catalog for `{catalog}`.`{schema}`.
Each entity lists its columns and Primary Keys.

Your ONLY task: analyze this catalog and identify EVERY foreign key relationship.
For each FK, specify which column in the child table references which PK in the parent table.

RULES:
1. Every FK MUST reference an existing PK in the catalog.
2. Do NOT invent columns — only use columns that exist in the catalog.
3. Include cardinality (1:1, 1:N, M:N).
4. Use action-verb descriptions (e.g., "Each order belongs to one customer").
5. Ensure no cross-entity relationship is missed.
6. For associative (junction) tables, there should be at least 2 FK relationships."""

    json_tmpl = """{
  "relationships": [
    {
      "from_table": "child_entity",
      "from_column": "fk_column",
      "to_table": "parent_entity",
      "to_column": "pk_column",
      "cardinality": "1:N",
      "description_english": "Each X belongs to one Y"
    }
  ]
}"""

    return f"""{context}

SCHEMA CATALOG:
{compressed_catalog}

Return ONLY valid JSON:
{json_tmpl}

Return ONLY the JSON. No markdown fences. No extra text."""


def _infer_heuristic_fks(proposed_tables, existing_rels):
    """Deterministic FK detection: catch obvious column-name matches the LLM missed.

    Two matching strategies:
    1. EXACT match: column `X_id` matches table with PK `X_id`
       e.g. order.restaurant_id → restaurant.restaurant_id
    2. SUFFIX match: column `prefix_X_id` ends with PK `X_id`
       e.g. flight.origin_airport_id → airport.airport_id
       e.g. flight.destination_airport_id → airport.airport_id

    Uses a multi-valued PK index (column → [tables]) to handle schemas where
    multiple tables share the same PK column name.
    """
    from collections import defaultdict

    # Build PK index: pk_column_name → [table_names]  (only single-column PKs)
    pk_index = defaultdict(list)
    for t in proposed_tables:
        tname = t.get("table_name", "")
        pk_cols = [c["name"] for c in t.get("columns", []) if c.get("role", "").lower() == "pk"]
        if len(pk_cols) == 1:
            pk_index[pk_cols[0]].append(tname)

    # Build set of existing relationship keys for fast lookup
    existing_keys = set()
    for r in existing_rels:
        key = (r.get("from_table", ""), r.get("from_column", ""), r.get("to_table", ""), r.get("to_column", ""))
        existing_keys.add(key)

    # Build valid column sets per table
    valid_cols = {t.get("table_name", ""): {c["name"] for c in t.get("columns", [])} for t in proposed_tables}

    def _try_add(tname, cname, parent_table, parent_pk, match_type):
        """Helper: add relationship if not already present and columns valid."""
        key = (tname, cname, parent_table, parent_pk)
        if key in existing_keys:
            return False
        if cname not in valid_cols.get(tname, set()):
            return False
        if parent_pk not in valid_cols.get(parent_table, set()):
            return False
        inferred.append({
            "from_table": tname,
            "from_column": cname,
            "to_table": parent_table,
            "to_column": parent_pk,
            "cardinality": "1:N",
            "description_english": f"Each {tname} record references one {parent_table} record via {cname}"
        })
        existing_keys.add(key)
        print(f"[HEURISTIC] ✅ {match_type}: {tname}.{cname} → {parent_table}.{parent_pk}", flush=True)
        return True

    inferred = []
    for t in proposed_tables:
        tname = t.get("table_name", "")
        for c in t.get("columns", []):
            cname = c.get("name", "")
            role = c.get("role", "").lower()
            if role == "pk":
                continue

            # --- Strategy 1: EXACT match (order_id matches orders.order_id) ---
            if cname.endswith("_id") and cname in pk_index:
                for parent_table in pk_index[cname]:
                    if parent_table != tname:
                        _try_add(tname, cname, parent_table, cname, "EXACT")

            # --- Strategy 2: SUFFIX match (origin_airport_id ends with airport_id) ---
            # Only if exact match didn't fire. Check if column ends with any PK name.
            if cname.endswith("_id") and cname not in pk_index:
                for pk_col, parent_tables in pk_index.items():
                    if not pk_col.endswith("_id"):
                        continue
                    # Column must end with _<pk_col> (e.g. origin_airport_id ends with _airport_id)
                    # and have a prefix before the PK name
                    if cname.endswith("_" + pk_col) and len(cname) > len(pk_col) + 1:
                        for parent_table in parent_tables:
                            if parent_table != tname:
                                _try_add(tname, cname, parent_table, pk_col, "SUFFIX")

    if inferred:
        print(f"[HEURISTIC] Added {len(inferred)} relationships via column-name matching", flush=True)
    else:
        print(f"[HEURISTIC] No additional relationships found (LLM covered all patterns)", flush=True)
    return inferred



def _rescue_orphans_via_llm(proposed_tables, existing_rels, token, model_type="3nf"):
    """LLM rescue pass: use Sonnet to find relationships for orphaned entities.

    After the Opus Reduce and deterministic heuristic, some entities may still
    have ZERO relationships — especially those with non-standard naming
    (e.g. origin_airport instead of airport_id, airline_code instead of airline_id,
    SAP-style Vendor_Code instead of vendor_id).

    This function:
    1. Identifies orphaned entities (entities with 0 incoming or outgoing rels)
    2. Builds a MINI catalog containing only orphans + a PK reference index
    3. Sends to Sonnet (fast, cheap) for targeted relationship inference
    4. Validates and returns new relationships

    Much smaller/cheaper than the full Reduce — typically <5K chars prompt.
    """
    # Find entities that participate in at least one relationship
    connected = set()
    for r in existing_rels:
        connected.add(r.get("from_table", ""))
        connected.add(r.get("to_table", ""))

    # Orphans = entities with ZERO relationships
    all_table_names = {t.get("table_name", "") for t in proposed_tables}
    orphan_names = all_table_names - connected
    if not orphan_names:
        print("[RESCUE] No orphaned entities — all connected", flush=True)
        return []

    print(f"[RESCUE] Found {len(orphan_names)} orphaned entities: {', '.join(sorted(orphan_names)[:10])}{'...' if len(orphan_names) > 10 else ''}", flush=True)

    # Build mini-catalog: full details for orphans, PK-only for connected entities
    orphan_tables = [t for t in proposed_tables if t.get("table_name", "") in orphan_names]
    connected_tables = [t for t in proposed_tables if t.get("table_name", "") in connected]

    lines = ["ORPHANED ENTITIES (need relationships):"]
    for t in orphan_tables:
        tname = t.get("table_name", "unknown")
        cols = t.get("columns", [])
        pk_cols = [c["name"] for c in cols if c.get("role", "").lower() == "pk"]
        other_cols = [f'{c["name"]} ({c.get("data_type", "STRING")})' for c in cols if c.get("role", "").lower() != "pk"]
        lines.append(f"  Entity: {tname}")
        if pk_cols:
            lines.append(f"    PK: {', '.join(pk_cols)}")
        if other_cols:
            lines.append(f"    Columns: {', '.join(other_cols)}")

    lines.append("")
    lines.append("CONNECTED ENTITIES (potential parents/children — PK only):")
    for t in connected_tables:
        tname = t.get("table_name", "unknown")
        cols = t.get("columns", [])
        pk_cols = [c["name"] for c in cols if c.get("role", "").lower() == "pk"]
        all_col_names = [c["name"] for c in cols]
        if pk_cols:
            lines.append(f"  Entity: {tname}  PK: {', '.join(pk_cols)}  Cols: {', '.join(all_col_names)}")

    mini_catalog = "\n".join(lines)
    print(f"[RESCUE] Mini-catalog: {len(mini_catalog):,} chars for {len(orphan_names)} orphans + {len(connected_tables)} connected", flush=True)

    # Build focused prompt
    prompt = f"""You are an expert data architect. I have entities that are DISCONNECTED (no relationships).
Your task: find FK→PK relationships that connect them to the rest of the model.

Look for SEMANTIC matches, not just naming conventions. Examples:
- origin_airport / destination_airport → references airport entity (by airport_code or airport_id)
- airline_code / carrier_code → references airline entity
- Vendor_Code → references vendor entity
- flight_id in bookings → references flight entity
- Company_Code → references company entity

{mini_catalog}

For EACH orphaned entity, identify ALL FK relationships to connected entities.
If an orphaned entity can also be a PARENT for another orphan, include that too.

Return ONLY valid JSON:
{{
  "relationships": [
    {{
      "from_table": "child_entity",
      "from_column": "fk_column",
      "to_table": "parent_entity",
      "to_column": "pk_column",
      "cardinality": "1:N",
      "description_english": "Each X references one Y"
    }}
  ]
}}

Return ONLY the JSON. No markdown fences. No extra text."""

    print(f"[RESCUE] Prompt: {len(prompt):,} chars", flush=True)

    # Use Sonnet (fast, cheap) — not Opus
    try:
        _refresh_endpoints_if_stale()
        pools = _get_model_pools()
        rescue_chain = pools["map"]["primary"] + pools["map"]["fallback"]
        raw = _call_chat_model(prompt, token, max_tokens=8000, model_chain=rescue_chain)
    except Exception as e:
        print(f"[RESCUE] ⚠ LLM call failed: {e}", flush=True)
        return []

    # Parse and validate
    parsed = _parse_llm_json_relationships(raw)
    rescue_rels = parsed.get("relationships", []) if parsed else []

    valid_tables = {t.get("table_name", ""): {c["name"] for c in t.get("columns", [])} for t in proposed_tables}
    existing_keys = set()
    for r in existing_rels:
        existing_keys.add((r.get("from_table", ""), r.get("from_column", ""), r.get("to_table", ""), r.get("to_column", "")))

    validated = []
    for r in rescue_rels:
        ft, fc = r.get("from_table", ""), r.get("from_column", "")
        tt, tc = r.get("to_table", ""), r.get("to_column", "")
        key = (ft, fc, tt, tc)
        if key in existing_keys:
            continue  # Already exists
        if ft in valid_tables and tt in valid_tables:
            if fc in valid_tables[ft] and tc in valid_tables[tt]:
                validated.append(r)
                existing_keys.add(key)
                print(f"[RESCUE] ✅ {ft}.{fc} → {tt}.{tc}", flush=True)
            else:
                print(f"[RESCUE] ⚠ Dropped: {ft}.{fc} → {tt}.{tc} (column not found)", flush=True)
        else:
            print(f"[RESCUE] ⚠ Dropped: {ft} → {tt} (table not found)", flush=True)

    print(f"[RESCUE] ✅ {len(validated)} new relationships rescued via Sonnet", flush=True)
    return validated


def _map_global_relationships(proposed_tables, catalog, schema, token, model_type="3nf", progress_cb=None):
    """Reduce phase: Use best Opus to infer all FK→PK relationships globally.

    Takes the merged Map output (all proposed entities), compresses it
    into a structure-only catalog, and sends to the best Opus endpoint
    (single call — no racing to save cost).
    Returns the relationships list and injects FK roles into columns.
    """
    _step = progress_cb or (lambda msg, **kw: None)

    # Build compressed catalog
    catalog_text = _build_schema_catalog(proposed_tables)
    print(f"[REDUCE] Schema catalog: {len(catalog_text):,} chars for {len(proposed_tables)} entities", flush=True)

    # Build prompt
    prompt = _build_reduce_prompt(catalog_text, catalog, schema, model_type)
    print(f"[REDUCE] Prompt: {len(prompt):,} chars", flush=True)

    # Single Opus call (best available — no racing to save cost)
    pools = _get_model_pools()
    reduce_models = pools["reduce"]["primary"][:1] + pools["reduce"]["fallback"]
    _short = lambda m: m.replace("databricks-claude-", "")
    _step(f"🧠 Calling {_short(reduce_models[0])} for global FK→PK mapping...")
    raw = _call_chat_model(prompt, token, max_tokens=MAX_TOKENS_SINGLE, model_chain=reduce_models)
    _step(f"🏆 {_short(reduce_models[0])} responded — {len(raw):,} chars")

    # Parse relationships
    _step("🔍 Parsing relationship JSON...")
    parsed = _parse_llm_json_relationships(raw)
    rels = parsed.get("relationships", []) if parsed else []

    # Validate: drop relationships pointing to non-existent tables/columns
    _step(f"✅ Validating {len(rels)} proposed relationships against entity schema...")
    valid_tables = {t.get("table_name", ""): {c["name"] for c in t.get("columns", [])} for t in proposed_tables}
    validated = []
    dropped = 0
    for r in rels:
        ft = r.get("from_table", "")
        fc = r.get("from_column", "")
        tt = r.get("to_table", "")
        tc = r.get("to_column", "")
        if ft in valid_tables and tt in valid_tables:
            if fc in valid_tables[ft] and tc in valid_tables[tt]:
                validated.append(r)
            else:
                dropped += 1
                print(f"[REDUCE] ⚠ Dropped rel: {ft}.{fc} → {tt}.{tc} (column not found)", flush=True)
        else:
            dropped += 1
            print(f"[REDUCE] ⚠ Dropped rel: {ft} → {tt} (table not found)", flush=True)

    print(f"[REDUCE] ✅ {len(validated)} valid relationships (dropped {dropped})", flush=True)
    _step(f"✅ {len(validated)} valid LLM relationships ({dropped} dropped)", status="done")

    # Heuristic FK detection: catch obvious column-name matches the LLM missed
    _step("🔗 Running heuristic FK detection (exact + suffix matching)...")
    heuristic_rels = _infer_heuristic_fks(proposed_tables, validated)
    if heuristic_rels:
        validated.extend(heuristic_rels)
        print(f"[REDUCE] ✅ Total after heuristic: {len(validated)} relationships", flush=True)
        _step(f"🔗 Heuristic found {len(heuristic_rels)} additional FK matches", status="done")
    else:
        _step("🔗 Heuristic: no additional matches (LLM caught them all)", status="done")

    # LLM rescue pass: use Sonnet to find relationships for orphaned entities
    # (entities with 0 relationships after Opus + heuristic — catches semantic
    # patterns like origin_airport→airport, airline_code→airline, Vendor_Code→vendor)
    _step("🔍 Scanning for orphaned entities...")
    try:
        rescue_rels = _rescue_orphans_via_llm(proposed_tables, validated, token, model_type)
        if rescue_rels:
            validated.extend(rescue_rels)
            print(f"[REDUCE] ✅ Total after rescue: {len(validated)} relationships", flush=True)
            _step(f"🚑 Rescue pass connected {len(rescue_rels)} orphaned entities", status="done")
        else:
            _step("🔍 No orphaned entities found — all connected", status="done")
    except Exception as rescue_err:
        print(f"[RESCUE] ⚠ Rescue pass failed (non-fatal): {rescue_err}", flush=True)
        _step(f"⚠ Rescue pass failed: {str(rescue_err)[:60]} (non-fatal)", status="error")

    return validated


def _parse_llm_json_relationships(raw):
    """Parse LLM response JSON specifically for the Reduce phase relationships output."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
    return data


# =============================================================================
# LLM ANALYSIS PIPELINE
# =============================================================================

def _analyze_single_pass(payload, catalog, schema, token, model_type="3nf"):
    """For small schemas: single Sonnet call with fallback chain. No racing = zero waste."""
    pools = _get_model_pools()
    model_chain = pools["map"]["primary"] + pools["map"]["fallback"]
    _short = lambda m: m.replace("databricks-claude-", "")
    ep_names = [_short(m) for m in model_chain]
    prompt = _build_analysis_prompt(payload, catalog, schema, model_type)
    table_count = payload.count(chr(10) + "Table: ")
    print(f"[LLM] Direct call: {len(prompt):,} chars for {table_count} tables → {ep_names[0]} (fallback: {', '.join(ep_names[1:])})", flush=True)
    _add_step(f"⏳ Analyzing {table_count} tables via {ep_names[0]} (fallback: {', '.join(ep_names[1:])})")

    _t0 = time.time()
    # Timer thread — updates modal every 10s so user sees progress
    _done = threading.Event()
    def _timer():
        while not _done.is_set():
            _done.wait(10)
            if not _done.is_set():
                elapsed = time.time() - _t0
                _update_step(f"⏳ Waiting for {ep_names[0]}... {elapsed:.0f}s elapsed (prompt: {len(prompt):,} chars)")
    threading.Thread(target=_timer, daemon=True).start()

    try:
        result = _call_chat_model(prompt, token, MAX_TOKENS_SINGLE, model_chain=model_chain)
    finally:
        _done.set()

    elapsed = time.time() - _t0
    _add_step(f"✅ Analysis complete — {len(result):,} chars in {elapsed:.0f}s via {ep_names[0]}", status="done")
    print(f"[LLM] ✅ {ep_names[0]} responded: {len(result):,} chars in {elapsed:.1f}s", flush=True)
    return result


def _analyze_groups(payload_per_table, catalog, schema, token, all_table_names, group_size=None, model_type="3nf"):
    """Distribute table groups across Sonnet endpoints in parallel (Map phase).

    group_size: dynamic — calculated to maximize parallelism across endpoints.
    Each group assigned to a specific Sonnet endpoint (round-robin).
    Sonnet: fast, cheap — ideal for structured entity extraction.
    Opus reserved for the global Reduce phase.
    """
    _short = lambda m: m.replace("databricks-claude-", "")
    pools = _get_model_pools()
    work_models = pools["map"]["primary"]
    map_fallback = pools["map"]["fallback"]
    n_eps = len(work_models) if work_models else 1

    # Dynamic group size: spread evenly across endpoints for max parallelism
    import math
    _gs = group_size or max(1, min(LLM_GROUP_SIZE, math.ceil(len(payload_per_table) / n_eps)))
    tables = list(payload_per_table.keys())
    groups = [tables[i:i + _gs] for i in range(0, len(tables), _gs)]

    # Log distribution plan — show each table's endpoint assignment
    ep_assignments = {}
    for gi, grp in enumerate(groups):
        ep = work_models[gi % n_eps]
        ep_name = _short(ep)
        if ep_name not in ep_assignments:
            ep_assignments[ep_name] = []
        ep_assignments[ep_name].extend(grp)

    # Show per-table assignments in log
    for gi, grp in enumerate(groups):
        ep = work_models[gi % n_eps]
        table_list = ", ".join(grp)
        print(f"[DIST] {table_list} → {_short(ep)}", flush=True)

    # Show per-endpoint summary in modal
    lines = []
    for ep_name, tbls in ep_assignments.items():
        lines.append(f"{ep_name}: {', '.join(tbls)}")
    _add_step("📋 " + " | ".join(lines))
    _update_step(f"⚡ {len(groups)} parallel calls dispatched", status="done")

    group_results = []
    group_errors = []
    completed = 0
    _groups_t0 = time.time()

    def _analyze_one_group(group_tables, endpoint_idx):
        ep = work_models[endpoint_idx % n_eps]
        # Fallback: other Sonnet first, then Opus, then Haiku
        other_primary = [m for m in work_models if m != ep]
        fallback = other_primary + map_fallback
        _g_t0 = time.time()
        group_payload = "\n".join(payload_per_table[t] for t in group_tables if t in payload_per_table)
        prompt = _build_group_prompt(group_payload, catalog, schema, all_table_names, model_type)
        print(f"[GROUP] ▶ Group {endpoint_idx+1} started → {_short(ep)} | {len(group_tables)} tables | prompt={len(prompt):,} chars", flush=True)

        try:
            result = _call_chat_model(prompt, token, MAX_TOKENS_GROUP, model_chain=[ep] + fallback)
        except OverflowError as oe:
            # Payload too large for the group — split into individual table calls
            print(f"[GROUP] ⚠ Group {endpoint_idx+1} too large ({len(prompt):,} chars). Splitting into {len(group_tables)} individual calls...", flush=True)
            sub_results = []
            for tbl in group_tables:
                if tbl not in payload_per_table:
                    continue
                sub_prompt = _build_group_prompt(payload_per_table[tbl], catalog, schema, all_table_names, model_type)
                try:
                    sub_result = _call_chat_model(sub_prompt, token, MAX_TOKENS_GROUP, model_chain=[ep] + fallback)
                    sub_results.append(sub_result)
                    print(f"[GROUP] ✓ Split table {tbl} done ({len(sub_result):,} chars)", flush=True)
                except Exception as sub_e:
                    print(f"[GROUP] ✗ Split table {tbl} failed: {sub_e}", flush=True)
            # Merge sub-results into a single JSON response
            if sub_results:
                merged_tables = []
                for sr in sub_results:
                    parsed = _parse_llm_json(sr, model_type)
                    if parsed:
                        merged_tables.extend(parsed.get("proposed_tables", []))
                result = json.dumps({"proposed_tables": merged_tables, "relationships": [], "columns_dropped": []})
                print(f"[GROUP] ✓ Group {endpoint_idx+1} recovered via split — {len(merged_tables)} tables", flush=True)
            else:
                raise RuntimeError(f"Group {endpoint_idx+1} failed: all split calls failed after payload overflow")

        elapsed = time.time() - _g_t0
        print(f"[GROUP] ✓ Group {endpoint_idx+1} done → {_short(ep)} | {len(result):,} chars | {elapsed:.1f}s", flush=True)
        return result, endpoint_idx

    _add_step(f"⏳ {len(groups)} Sonnet calls running in parallel (Map phase)...")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        futures = {
            pool.submit(_analyze_one_group, grp, i): i
            for i, grp in enumerate(groups)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_text, gidx = future.result()
                if result_text:
                    parsed = _parse_llm_json(result_text, model_type)
                    if parsed:
                        n_tables = len(parsed.get("proposed_tables", []))
                        group_results.append(parsed)
                        print(f"[GROUP] ✓ Group {gidx+1} parsed: {n_tables} tables proposed", flush=True)
            except Exception as e:
                ep = work_models[idx % n_eps] if work_models else "?"
                group_errors.append(f"Group {idx+1} ({_short(ep)}): {str(e)[:80]}")
                print(f"[GROUP] ✗ Group {idx+1} FAILED ({_short(ep)}): {e}", flush=True)
            completed += 1
            elapsed = time.time() - _groups_t0
            # Show per-table completion
            grp_tables = groups[idx] if idx < len(groups) else []
            tbl_name = grp_tables[0] if len(grp_tables) == 1 else f"{len(grp_tables)} tables"
            if not group_errors:
                _update_step(f"✅ {completed}/{len(groups)} done — {tbl_name} complete ({elapsed:.0f}s)")
            else:
                _update_step(f"⏳ {completed}/{len(groups)} — {len(group_results)} OK, {len(group_errors)} failed ({elapsed:.0f}s)")

    total_elapsed = time.time() - _groups_t0
    total_proposed = sum(len(g.get("proposed_tables", [])) for g in group_results)
    _update_step(f"All {len(groups)} groups done — {total_proposed} tables proposed, {len(group_errors)} errors ({total_elapsed:.0f}s)", status="done")

    if group_errors:
        for err in group_errors[:3]:
            _add_step(f"⚠ {err}", status="error")
        print(f"[GROUP] {len(group_errors)} group(s) failed: {group_errors}", flush=True)

    if not group_results:
        raise RuntimeError(f"All {len(groups)} groups failed: {'; '.join(group_errors[:3])}")

    # Merge — PROGRAMMATIC (instant, no LLM call needed)
    # Each per-table call already knows all table names, so proposals are
    # cross-referenced. We just concatenate + deduplicate in Python.
    _merge_t0 = time.time()
    _add_step(f"🔗 Merging {len(group_results)} proposals ({total_proposed} tables)...")

    merged_tables = {}   # table_name -> table dict (dedup by name, keep richest)
    merged_rels = []     # all relationships (dedup later)
    merged_dropped = []
    merged_summaries = []

    for g in group_results:
        # Merge proposed tables (deduplicate by table_name, keep the one with more columns)
        for t in g.get("proposed_tables", []):
            tname = t.get("table_name", "")
            if tname not in merged_tables or len(t.get("columns", [])) > len(merged_tables[tname].get("columns", [])):
                merged_tables[tname] = t

        # Collect all relationships
        for r in g.get("relationships", []):
            merged_rels.append(r)

        # Collect dropped columns
        for d in g.get("columns_dropped", []):
            merged_dropped.append(d)

        # Collect summaries
        s = g.get("data_summary", "")
        if s:
            merged_summaries.append(s)


    # Deduplicate relationships by (from_table, from_column, to_table, to_column)
    seen_rels = set()
    unique_rels = []
    for r in merged_rels:
        key = (r.get("from_table", ""), r.get("from_column", ""), r.get("to_table", ""), r.get("to_column", ""))
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(r)

    # Deduplicate dropped columns
    seen_dropped = set()
    unique_dropped = []
    for d in merged_dropped:
        key = (d.get("source_table", ""), d.get("column", ""))
        if key not in seen_dropped:
            seen_dropped.add(key)
            unique_dropped.append(d)

    # Build unified model
    unified = {
        "proposed_tables": list(merged_tables.values()),
        "relationships": unique_rels,
        "columns_dropped": unique_dropped,
        "data_summary": " ".join(merged_summaries[:3]),  # Combine first 3 summaries
    }

    merge_elapsed = time.time() - _merge_t0
    t_count = len(unified["proposed_tables"])
    r_count = len(unified["relationships"])
    print(f"[MERGE] ✓ Programmatic merge: {t_count} tables, {r_count} rels in {merge_elapsed*1000:.0f}ms", flush=True)
    _update_step(f"✅ Merged — {t_count} tables, {r_count} relationships (instant)", status="done")
    return json.dumps(unified)


# =============================================================================
# JSON PARSING & VALIDATION
# =============================================================================

def _parse_llm_json(raw, model_type="3nf"):
    """Parse LLM response JSON with tolerance for markdown fences."""
    if not raw:
        return None

    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try fixing common issues
        try:
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

    # Validate required keys
    if "proposed_tables" not in data:
        return None

    # Normalize table types (3NF entity types)
    if model_type == "dimensional":
        valid_types = {"fact", "dimension", "bridge", "aggregate"}
    else:
        valid_types = {"core_entity", "weak_entity", "associative", "reference"}
    for t in data.get("proposed_tables", []):
        tt = t.get("table_type", "").lower()
        if tt not in valid_types:
            name = t.get("table_name", "").lower()
            if model_type == "dimensional":
                # Dimensional fallback
                if "fact" in name or "fct" in name:
                    t["table_type"] = "fact"
                elif "bridge" in name or "link" in name or "junction" in name:
                    t["table_type"] = "bridge"
                elif "agg" in name or "summary" in name:
                    t["table_type"] = "aggregate"
                else:
                    t["table_type"] = "dimension"
            else:
                # 3NF fallback
                if "assoc" in name or "bridge" in name or "link" in name or "junction" in name:
                    t["table_type"] = "associative"
                elif "ref" in name or "lkp" in name or "lookup" in name or "code" in name:
                    t["table_type"] = "reference"
                elif "_dep" in name or "detail" in name or "line" in name:
                    t["table_type"] = "weak_entity"
                else:
                    t["table_type"] = "core_entity"
        else:
            t["table_type"] = tt

        # Normalize column roles (3NF: no "measure" role)
        valid_roles = {"pk", "fk", "measure", "attribute"} if model_type == "dimensional" else {"pk", "fk", "attribute"}
        for c in t.get("columns", []):
            role = c.get("role", "").lower()
            if role not in valid_roles:
                c["role"] = "attribute"
            else:
                c["role"] = role

    # Ensure defaults
    data.setdefault("relationships", [])
    data.setdefault("columns_dropped", [])
    data.setdefault("data_summary", "")

    return data


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def _reset_job():
    global _job
    _job = {
        "active": True,
        "steps": [],
        "result": None,
        "error": None,
        "cancelled": False,
        "start_time": datetime.now(),
    }


def _add_step(msg, status="active"):
    ts = datetime.now().strftime("%H:%M:%S")
    # Mark previous active step as done
    for s in _job["steps"]:
        if s["status"] == "active":
            s["status"] = "done"
    _job["steps"].append({"time": ts, "status": status, "msg": msg})


def _update_step(msg, status=None):
    if _job["steps"]:
        _job["steps"][-1]["msg"] = msg
        if status:
            _job["steps"][-1]["status"] = status


def _check_cancelled():
    if _job.get("cancelled"):
        raise RuntimeError("Cancelled by user")


# =============================================================================
# BACKGROUND GENERATION THREAD
# =============================================================================

def _bg_generate(catalog, schema, model_type, token):
    """Background thread: profile schema, call LLM, parse result."""
    _tls.token = token
    try:
        # Step 1: Profile
        profile_result = _profile_schema(catalog, schema, token)
        if not profile_result:
            _job["error"] = "No tables found in schema"
            _job["active"] = False
            return
        raw_profiles, prof_catalog, prof_schema = profile_result

        _check_cancelled()

        # Step 2: LLM Analysis — Map phase uses Sonnet, Reduce phase uses Opus
        # Build per-table payloads DIRECTLY from profiles (no truncation)
        per_table = _build_per_table_payloads(raw_profiles, prof_catalog, prof_schema)
        table_count = len(per_table)
        total_chars = sum(len(v) for v in per_table.values())

        pools = _get_model_pools()
        map_models = pools["map"]["primary"]
        ep_names = [m.replace("databricks-claude-", "") for m in map_models]
        n_eps = len(map_models) if map_models else 1
        print(f"[GEN] Map phase: {table_count} tables → distributing across {n_eps} Sonnet endpoints, total payload={total_chars:,} chars", flush=True)

        if table_count == 0:
            _job["error"] = "No tables could be profiled"
            _add_step("❌ No tables profiled successfully", status="error")
            _job["active"] = False
            return

        if table_count == 1:
            # Edge case: only 1 table — direct call, no distribution needed
            single_payload = list(per_table.values())[0]
            _add_step(f"🧠 Analyzing 1 table via {ep_names[0]} (Sonnet Map)")
            raw_response = _analyze_single_pass(single_payload, prof_catalog, prof_schema, token, model_type)
        else:
            # Groups of LLM_GROUP_SIZE tables, round-robin across Sonnet endpoints
            _add_step(f"⚡ {table_count} tables in groups of {LLM_GROUP_SIZE} → Sonnet round-robin {', '.join(ep_names)}")
            raw_response = _analyze_groups(per_table, prof_catalog, prof_schema, token, list(per_table.keys()), model_type=model_type)

        _check_cancelled()

        # Step 3: Parse Map output
        _add_step(f"🔍 Parsing Map response ({len(raw_response):,} chars) — extracting entities...")
        model = _parse_llm_json(raw_response, model_type)
        if not model:
            _job["error"] = "Failed to parse LLM response as valid JSON"
            _add_step("❌ Parse failed — LLM returned invalid JSON", status="error")
            print(f"[PARSE] Failed. First 500 chars: {raw_response[:500]}", flush=True)
        else:
            t_count = len(model.get("proposed_tables", []))
            col_count = sum(len(t.get("columns", [])) for t in model.get("proposed_tables", []))
            _add_step(f"✅ Map complete — {t_count} entities, {col_count} columns extracted", status="done")
            print(f"[PARSE] ✅ Map: {t_count} tables, {col_count} cols", flush=True)

            _check_cancelled()

            # Step 4: Reduce — Global relationship mapping via Opus
            _add_step(f"🧠 Reduce phase — calling best Opus for global FK→PK mapping...")
            try:
                relationships = _map_global_relationships(
                    model.get("proposed_tables", []), catalog, schema, token, model_type,
                    progress_cb=_add_step
                )
                model["relationships"] = relationships

                # Inject FK roles into columns
                for rel in relationships:
                    for t in model.get("proposed_tables", []):
                        if t["table_name"] == rel.get("from_table", ""):
                            for c in t.get("columns", []):
                                if c["name"] == rel.get("from_column", ""):
                                    c["role"] = "fk"

                r_count = len(relationships)
                _add_step(f"✅ Reduce complete — {r_count} relationships mapped (Opus + heuristic + LLM rescue)", status="done")
                print(f"[REDUCE] ✅ {r_count} relationships injected", flush=True)
            except Exception as reduce_err:
                print(f"[REDUCE] ⚠ Reduce phase failed: {reduce_err}", flush=True)
                _add_step(f"⚠ Reduce failed: {str(reduce_err)[:80]} — model has entities but no relationships", status="error")
                model.setdefault("relationships", [])

            t_count = len(model.get("proposed_tables", []))
            r_count = len(model.get("relationships", []))
            col_count = sum(len(t.get("columns", [])) for t in model.get("proposed_tables", []))
            _add_step(f"✅ Model ready — {t_count} tables, {r_count} relationships, {col_count} columns", status="done")
            _job["result"] = model
            print(f"[FINAL] ✅ {t_count} tables, {r_count} rels, {col_count} cols", flush=True)

    except RuntimeError as e:
        if "Cancelled" in str(e):
            _add_step("Cancelled by user", status="error")
        else:
            _add_step(f"Error: {str(e)[:100]}", status="error")
        _job["error"] = str(e)
    except Exception as e:
        _add_step(f"Error: {str(e)[:100]}", status="error")
        _job["error"] = str(e)
        traceback.print_exc()
    finally:
        _job["active"] = False


# =============================================================================
# ERD VISUALIZATION BUILDER
# =============================================================================

def _detect_domain_clusters(tables, relationships):
    """Detect connected components (domain clusters) using BFS on relationships.

    Returns dict: table_name -> cluster_id (int starting from 0).
    Each connected component = one domain cluster.
    Isolated nodes (no relationships) get their own cluster.
    """
    # Build adjacency list
    adj = defaultdict(set)
    table_names = {t.get("table_name", "") for t in tables}
    for r in relationships:
        ft = r.get("from_table", "")
        tt = r.get("to_table", "")
        if ft in table_names and tt in table_names:
            adj[ft].add(tt)
            adj[tt].add(ft)

    # BFS to find connected components
    visited = set()
    clusters = {}  # table_name -> cluster_id
    cluster_id = 0

    for t in tables:
        tname = t.get("table_name", "")
        if tname in visited:
            continue
        # BFS from this node
        queue = [tname]
        component = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        for node in component:
            clusters[node] = cluster_id
        cluster_id += 1

    return clusters, cluster_id


# Domain cluster label seeds — try to name clusters by their dominant entity
_DOMAIN_KEYWORDS = {
    "flight": "Aviation", "airline": "Aviation", "airport": "Aviation", "booking": "Aviation",
    "passenger": "Aviation", "ticket": "Aviation",
    "order": "Orders", "restaurant": "Orders", "delivery": "Orders", "payment": "Orders",
    "menu": "Orders", "driver": "Orders", "food": "Orders",
    "procurement": "Procurement", "contract": "Procurement", "vendor": "Procurement",
    "invoice": "Procurement", "purchase": "Procurement",
    "customer": "Customers", "user": "Customers", "account": "Customers",
    "product": "Products", "item": "Products", "inventory": "Products",
    "employee": "HR", "department": "HR", "salary": "HR",
    "agent": "Agent/ML", "evaluation": "Agent/ML", "model": "Agent/ML",
    "trace": "Agent/ML", "metric": "Agent/ML",
}


def _name_cluster(table_names):
    """Try to give a domain-meaningful name to a cluster based on its table names."""
    # Count domain keyword matches
    domain_votes = defaultdict(int)
    for tname in table_names:
        for keyword, domain in _DOMAIN_KEYWORDS.items():
            if keyword in tname.lower():
                domain_votes[domain] += 1
    if domain_votes:
        return max(domain_votes, key=domain_votes.get)
    # Fall back to first table name
    return table_names[0] if table_names else "cluster"


def build_proposed_erd_elements(model):
    """Build Cytoscape elements from LLM model proposal with domain clustering.

    Uses connected component detection to group related entities into
    compound parent nodes. This makes fcose layout place each domain
    cluster together and separate unrelated domains visually.
    """
    elements = []
    if not model:
        return elements

    tables = model.get("proposed_tables", [])
    relationships = model.get("relationships", [])

    # Detect domain clusters (connected components)
    clusters, num_clusters = _detect_domain_clusters(tables, relationships)

    # Build cluster -> table_names mapping
    cluster_tables = defaultdict(list)
    for t in tables:
        tname = t.get("table_name", "")
        cid = clusters.get(tname, 0)
        cluster_tables[cid].append(tname)

    # Only create compound parents if there are multiple clusters with >1 node
    multi_node_clusters = {cid: names for cid, names in cluster_tables.items() if len(names) > 1}
    use_compound = len(multi_node_clusters) > 1 or (len(multi_node_clusters) == 1 and len(cluster_tables) > len(multi_node_clusters))

    # Add compound parent nodes for each cluster (if clustering is useful)
    cluster_parent_ids = {}
    if use_compound:
        for cid, table_names in cluster_tables.items():
            parent_id = f"__cluster_{cid}"
            cluster_parent_ids[cid] = parent_id
            domain_name = _name_cluster(table_names)
            elements.append({
                "data": {
                    "id": parent_id,
                    "label": f"{domain_name} ({len(table_names)})" if len(table_names) > 1 else "",
                },
                "classes": "compound-node",
            })

    # Nodes — one per proposed table
    for t in tables:
        tname = t.get("table_name", "unknown")
        ttype = t.get("table_type", "core_entity").lower()
        col_count = len(t.get("columns", []))
        color = COLORS.get(ttype, COLORS["attribute"])

        node_data = {
            "id": tname,
            "label": f"{tname}\n({col_count} cols)",
            "table_type": ttype,
            "color": color,
            "description": t.get("description", ""),
            "columns": json.dumps(t.get("columns", [])),
            "source_tables": json.dumps(t.get("source_tables", [])),
        }

        # Assign to compound parent if clustering is active
        cid = clusters.get(tname, 0)
        if use_compound and cid in cluster_parent_ids:
            node_data["parent"] = cluster_parent_ids[cid]

        elements.append({
            "data": node_data,
            "classes": ttype,
        })

    # Edges — consolidated: ONE per table pair
    edge_map = defaultdict(list)  # (from_table, to_table) -> [relationships]
    for r in relationships:
        ft = r.get("from_table", "")
        tt = r.get("to_table", "")
        if ft and tt:
            key = tuple(sorted([ft, tt]))
            edge_map[key].append(r)

    for (t1, t2), rels in edge_map.items():
        # Use first relationship's direction
        first = rels[0]

        # Build label: cardinality + column names on the arrow
        # e.g. "1:N  order_id" or "1:N  origin_airport_id → airport_id"
        label_parts = []
        for r in rels:
            fc = r.get("from_column", "")
            tc = r.get("to_column", "")
            card = r.get("cardinality", "")
            if fc and tc:
                if fc == tc:
                    label_parts.append(f"{card}  {fc}")
                else:
                    label_parts.append(f"{card}  {fc} → {tc}")
            else:
                label_parts.append(card)
        label = "\n".join(label_parts)

        elements.append({
            "data": {
                "id": f"edge_{first.get('from_table', 'x')}_{first.get('to_table', 'y')}",
                "source": first.get("from_table", ""),
                "target": first.get("to_table", ""),
                "label": label,
                "description": first.get("description_english", ""),
                "rel_count": len(rels),
                "all_rels": json.dumps(rels),
            }
        })

    return elements


def get_cytoscape_stylesheet():
    """Get the Cytoscape stylesheet for the ERD graph."""
    return [
        # Default node
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "11px",
                "color": "#ffffff",
                "background-color": "data(color)",
                "shape": "round-rectangle",
                "width": 200,
                "height": 56,
                "border-width": 2,
                "border-color": "#cbd5e1",
                "text-max-width": 180,
            },
        },
        # 3NF Entity types
        {"selector": ".core_entity", "style": {"background-color": COLORS["core_entity"], "border-color": "#1e40af"}},
        {"selector": ".weak_entity", "style": {"background-color": COLORS["weak_entity"], "border-color": "#0284c7"}},
        {"selector": ".associative", "style": {"background-color": COLORS["associative"], "border-color": "#ea580c"}},
        {"selector": ".reference", "style": {"background-color": COLORS["reference"], "border-color": "#475569"}},
        # Dimensional Entity types
        {"selector": ".fact", "style": {"background-color": COLORS["fact"], "border-color": "#1d4ed8"}},
        {"selector": ".dimension", "style": {"background-color": COLORS["dimension"], "border-color": "#15803d"}},
        {"selector": ".bridge", "style": {"background-color": COLORS["bridge"], "border-color": "#c2410c"}},
        {"selector": ".aggregate", "style": {"background-color": COLORS["aggregate"], "border-color": "#6d28d9"}},
        # Edges
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#94a3b8",
                "line-color": "#475569",
                "width": 2,
                "font-size": "8px",
                "font-family": "'SF Mono', 'Cascadia Code', 'Fira Code', monospace",
                "color": "#64748b",
                "text-wrap": "wrap",
                "text-max-width": 200,
                "text-rotation": "autorotate",
                "text-margin-y": -10,
                "text-background-color": "#f8fafc",
                "text-background-opacity": 0.85,
                "text-background-padding": "3px",
                "text-background-shape": "roundrectangle",
            },
        },
        # Selected/hovered
        {
            "selector": "node:selected",
            "style": {
                "border-width": 4,
                "border-color": "#facc15",
            },
        },
        # Dimmed (for search filter)
        {
            "selector": ".dimmed",
            "style": {
                "opacity": 0.2,
            },
        },
        # Compound parent nodes (domain clusters)
        {
            "selector": ".compound-node",
            "style": {
                "background-color": "#f1f5f9",
                "background-opacity": 0.65,
                "border-width": 2,
                "border-color": "#cbd5e1",
                "border-style": "dashed",
                "shape": "round-rectangle",
                "padding": "30px",
                "text-valign": "top",
                "text-halign": "center",
                "font-size": "13px",
                "font-weight": "bold",
                "color": "#64748b",
                "text-margin-y": 8,
                "label": "data(label)",
            },
        },
    ]


# =============================================================================
# SUMMARY PANEL BUILDER
# =============================================================================

def build_summary_panel(model):
    mt = model.get("_model_type", "3nf")
    """Build full-width summary section below ERD with card layout."""
    if not model:
        return html.Div("No model generated yet.", style={"color": "#94a3b8", "padding": "20px"})

    sections = []

    # --- Business Domain Overview card ---
    summary = model.get("data_summary", "")
    if summary:
        sections.append(
            html.Div([
                html.H3(["\U0001f4cb Business Domain Overview"], className="card-title"),
                html.P(summary, style={"color": "#334155", "lineHeight": "1.8", "fontSize": "14px"}),
            ], className="summary-card")
        )



    # --- Proposed Tables card (HTML table) ---
    tables = model.get("proposed_tables", [])
    if tables:
        header = html.Tr([
            html.Th("TYPE", style={"width": "100px"}),
            html.Th("TABLE NAME", style={"width": "180px"}),
            html.Th("DESCRIPTION"),
            html.Th("COLS", style={"width": "60px", "textAlign": "center"}),
            html.Th("KEYS", style={"width": "120px"}),
            html.Th("SOURCE", style={"width": "200px"}),
        ], className="table-header-row")

        rows = []
        for t in tables:
            ttype = t.get("table_type", "fact" if mt == "dimensional" else "core_entity")
            cols = t.get("columns", [])
            pks = sum(1 for c in cols if c.get("role", "").lower() == "pk")
            fks = sum(1 for c in cols if c.get("role", "").lower() == "fk")
            attrs = sum(1 for c in cols if c.get("role", "").lower() == "attribute")
            measures = sum(1 for c in cols if c.get("role", "").lower() == "measure")
            sources = ", ".join(t.get("source_tables", []))
            rows.append(html.Tr([
                html.Td(html.Span(ttype.upper(), className=f"badge badge-{ttype}")),
                html.Td(t.get("table_name", "?"), style={"fontWeight": "600"}),
                html.Td(t.get("description", ""), style={"color": "#475569", "fontSize": "13px"}),
                html.Td(str(len(cols)), style={"textAlign": "center", "fontWeight": "600", "color": "#2563eb"}),
                html.Td(f"{pks}PK / {fks}FK / {measures}M / {attrs}A" if mt == "dimensional" else f"{pks}PK / {fks}FK / {attrs}A", style={"fontSize": "12px", "color": "#64748b"}),
                html.Td(sources, style={"fontSize": "12px", "color": "#64748b"}),
            ]))

        sections.append(
            html.Div([
                html.H3([f"\U0001f4ca Proposed Tables ({len(tables)})"], className="card-title"),
                html.Table([html.Thead(header), html.Tbody(rows)], className="data-table"),
            ], className="summary-card")
        )

    # --- Relationships card (HTML table) ---
    rels = model.get("relationships", [])
    if rels:
        rel_header = html.Tr([
            html.Th("FROM TABLE", style={"width": "180px"}),
            html.Th("COLUMN", style={"width": "200px"}),
            html.Th("", style={"width": "30px", "textAlign": "center"}),
            html.Th("TO TABLE", style={"width": "180px"}),
            html.Th("COLUMN", style={"width": "200px"}),
            html.Th("RELATIONSHIP"),
        ], className="table-header-row")

        rel_rows = []
        for r in rels:
            rel_rows.append(html.Tr([
                html.Td(r.get("from_table", "?"), style={"fontWeight": "500"}),
                html.Td(r.get("from_column", ""), style={"color": "#2563eb"}),
                html.Td("\u2192", style={"textAlign": "center", "color": "#94a3b8"}),
                html.Td(r.get("to_table", "?"), style={"fontWeight": "500"}),
                html.Td(r.get("to_column", ""), style={"color": "#2563eb"}),
                html.Td(r.get("description_english", ""), style={"color": "#64748b", "fontStyle": "italic", "fontSize": "13px"}),
            ]))

        sections.append(
            html.Div([
                html.H3([f"\U0001f517 Relationships ({len(rels)})"], className="card-title"),
                html.Table([html.Thead(rel_header), html.Tbody(rel_rows)], className="data-table"),
            ], className="summary-card")
        )

    # --- Dropped Columns (collapsible) ---
    dropped = model.get("columns_dropped", [])
    if dropped:
        drop_rows = [html.Tr([
            html.Td(d.get("source_table", "?"), style={"fontWeight": "500"}),
            html.Td(d.get("column", "?"), style={"color": "#64748b"}),
            html.Td(d.get("reason", ""), style={"color": "#94a3b8", "fontSize": "13px"}),
        ]) for d in dropped]

        sections.append(
            html.Details([
                html.Summary(f"\U0001f5d1 Dropped Columns ({len(dropped)})",
                             className="card-title", style={"cursor": "pointer", "padding": "16px 24px"}),
                html.Div(
                    html.Table([
                        html.Thead(html.Tr([html.Th("TABLE"), html.Th("COLUMN"), html.Th("REASON")])),
                        html.Tbody(drop_rows),
                    ], className="data-table"),
                    style={"padding": "0 24px 16px"},
                ),
            ], className="summary-card")
        )

    return html.Div(sections, id="summary-section")


# =============================================================================
# PDF REPORT GENERATION (Server-Side — Consulting Grade)
# =============================================================================

def _generate_consulting_report(model):
    """Generate a McKinsey-style consulting PDF report for the dimensional model."""
    from fpdf import FPDF
    import io

    def _s(text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        for uc, repl in {"\u2022":"-","\u2013":"-","\u2014":"-","\u2018":"'","\u2019":"'",
                         "\u201c":'"',"\u201d":'"',"\u2026":"...","\u2192":"->","\u00a0":" ",
                         "\u200b":"","\u20b9":"INR "}.items():
            text = text.replace(uc, repl)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def _deep(obj):
        if isinstance(obj, str): return _s(obj)
        if isinstance(obj, list): return [_deep(x) for x in obj]
        if isinstance(obj, dict): return {k: _deep(v) for k, v in obj.items()}
        return obj

    m = _deep(model)

    # Load Databricks logo path for PDF footer
    _db_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "db-log.png")

    class Report(FPDF):
        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(160, 160, 160)
            # Left: app name
            self.cell(90, 5, "AI Data Model Designer  |  Confidential", align="L")
            # Center: Powered by Databricks logo
            try:
                _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "db-log.png")
                cx = (self.w / 2) - 22
                self.set_xy(cx, self.h - 14)
                self.set_font("Helvetica", "", 7)
                self.set_text_color(160, 160, 160)
                self.cell(20, 5, "Powered by")
                self.image(_logo, x=cx + 20, y=self.h - 13.5, h=4)
            except Exception:
                pass
            # Right: page number
            self.set_xy(self.w - 18 - 20, self.h - 14)
            self.cell(20, 5, f"Page {self.page_no()}", align="R")

    pdf = Report("L", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    M = 18
    pw = 297
    W = pw - 2 * M
    NAVY = (10, 25, 47)
    BLUE = (0, 82, 155)
    DGRAY = (55, 55, 55)
    MGRAY = (120, 120, 120)
    LGRAY = (200, 200, 200)
    VLIGHT = (245, 245, 245)
    WHITE = (255, 255, 255)

    tables = m.get("proposed_tables", [])
    rels = m.get("relationships", [])
    dropped = m.get("columns_dropped", [])
    summary_text = m.get("data_summary", "")
    catalog = m.get("_catalog", "")
    schema = m.get("_schema", "")
    mt = m.get("_model_type", "3nf")
    total_cols = sum(len(t.get("columns", [])) for t in tables)

    if mt == "dimensional":
        type_order = {"fact": 0, "dimension": 1, "bridge": 2, "aggregate": 3}
        CLR = {"fact": (37, 99, 235), "dimension": (22, 163, 74), "bridge": (234, 88, 12), "aggregate": (124, 58, 237)}
    else:
        type_order = {"core_entity": 0, "weak_entity": 1, "associative": 2, "reference": 3}
        CLR = {"core_entity": (30, 58, 138), "weak_entity": (14, 165, 233), "associative": (249, 115, 22), "reference": (100, 116, 139)}

    tables_sorted = sorted(tables, key=lambda t: type_order.get(t.get("table_type", "").lower(), 9))
    type_counts = {}
    default_type = "fact" if mt == "dimensional" else "core_entity"
    for t in tables:
        tt = t.get("table_type", default_type).lower()
        type_counts[tt] = type_counts.get(tt, 0) + 1

    # --- McKinsey-style table helper ---
    def draw_table(headers, rows, col_widths, start_y=None, accent=BLUE):
        """Draw a clean consulting-style table with only horizontal rules."""
        if start_y:
            pdf.set_y(start_y)
        y = pdf.get_y()
        lh = 5.5  # line height
        hdr_h = 7

        # Header background - thin navy bar
        pdf.set_fill_color(*NAVY)
        pdf.rect(M, y, W, hdr_h, "F")
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*WHITE)
        x = M
        for i, h in enumerate(headers):
            cw = col_widths[i] if i < len(col_widths) else col_widths[-1]
            pdf.set_xy(x + 2, y + 1)
            pdf.cell(cw - 2, hdr_h - 2, h.upper())
            x += cw
        y += hdr_h

        # Data rows - alternating subtle background, no vertical lines
        pdf.set_font("Helvetica", "", 7.5)
        for ri, row_data in enumerate(rows):
            # Check page break
            if y + lh > 210 - 15:
                pdf.add_page()
                y = M
                # Repeat header
                pdf.set_fill_color(*NAVY)
                pdf.rect(M, y, W, hdr_h, "F")
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*WHITE)
                x = M
                for i, h in enumerate(headers):
                    cw = col_widths[i] if i < len(col_widths) else col_widths[-1]
                    pdf.set_xy(x + 2, y + 1)
                    pdf.cell(cw - 2, hdr_h - 2, h.upper())
                    x += cw
                y += hdr_h
                pdf.set_font("Helvetica", "", 7.5)

            # Alternate row fill
            if ri % 2 == 1:
                pdf.set_fill_color(*VLIGHT)
                pdf.rect(M, y, W, lh, "F")

            # Cell text
            pdf.set_text_color(*DGRAY)
            x = M
            for i, val in enumerate(row_data):
                cw = col_widths[i] if i < len(col_widths) else col_widths[-1]
                # Bold first column
                if i == 0:
                    pdf.set_font("Helvetica", "B", 7.5)
                else:
                    pdf.set_font("Helvetica", "", 7.5)
                pdf.set_xy(x + 2, y + 0.5)
                # Truncate to fit
                txt = str(val)[:int(cw / 1.6)]
                pdf.cell(cw - 2, lh - 1, txt)
                x += cw

            # Thin separator line
            pdf.set_draw_color(*LGRAY)
            pdf.set_line_width(0.1)
            pdf.line(M, y + lh, M + W, y + lh)
            y += lh

        # Bottom border - thicker
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        pdf.line(M, y, M + W, y)
        pdf.set_y(y + 3)
        return y

    def section_title(num, title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*NAVY)
        pdf.cell(W, 8, f"{num}   {title}", new_x="LMARGIN", new_y="NEXT")
        # Accent underline
        pdf.set_draw_color(*BLUE)
        pdf.set_line_width(0.6)
        pdf.line(M, pdf.get_y(), M + 60, pdf.get_y())
        pdf.ln(5)

    def body_text(text, sz=9):
        pdf.set_font("Helvetica", "", sz)
        pdf.set_text_color(*DGRAY)
        pdf.multi_cell(W, 4.5, text)
        pdf.ln(3)

    def label_value(label, value):
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MGRAY)
        lw = pdf.get_string_width(label + "  ") + 2
        pdf.cell(lw, 4.5, label)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*DGRAY)
        pdf.cell(0, 4.5, str(value), new_x="LMARGIN", new_y="NEXT")

    # ==========================================
    # COVER PAGE
    # ==========================================
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pw, 210, "F")

    # Thin accent line
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(1.5)
    pdf.line(M, 55, M + 80, 55)

    pdf.set_xy(M, 62)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*LGRAY)
    pdf.cell(W, 6, "DIMENSIONAL DATA MODEL" if mt == "dimensional" else "RELATIONAL DATA MODEL (3NF)")
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*WHITE)
    pdf.cell(W, 14, "Design Report")
    pdf.ln(18)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(150, 180, 210)
    if catalog and schema:
        pdf.cell(W, 7, f"{catalog}.{schema}")
        pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*LGRAY)
    pdf.cell(W, 5, datetime.now().strftime("%B %d, %Y"))
    pdf.ln(25)

    # Key metrics in a clean row
    metrics = [("TABLES", str(len(tables))), ("RELATIONSHIPS", str(len(rels))),
               ("COLUMNS", str(total_cols)), ("DROPPED", str(len(dropped)))]
    box_w = 45
    pdf.set_x(M)
    for label, val in metrics:
        x0 = pdf.get_x()
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(x0, pdf.get_y())
        pdf.cell(box_w, 12, val)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*LGRAY)
        pdf.set_xy(x0, pdf.get_y() + 12)
        pdf.cell(box_w, 5, label)
        pdf.set_xy(x0 + box_w + 5, pdf.get_y() - 12)

    # ==========================================
    # EXECUTIVE SUMMARY
    # ==========================================
    pdf.add_page()
    section_title("01", "Business Domain Overview")
    body_text(summary_text or "No business domain overview available.", 9)


    # Type distribution
    if type_counts:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(W, 5, "Model Composition", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for tt, cnt in sorted(type_counts.items(), key=lambda x: type_order.get(x[0], 9)):
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*DGRAY)
            pdf.cell(W, 4.5, f"    {tt.title()}: {cnt} table{'s' if cnt > 1 else ''}", new_x="LMARGIN", new_y="NEXT")

    # ==========================================
    # DATA MODEL OVERVIEW
    # ==========================================
    pdf.add_page()
    section_title("02", "Data Model Overview")
    body_text("The following table inventory summarizes all proposed tables in the 3NF relational model, "
              "including type classification, column counts, and source lineage.", 8)
    pdf.ln(2)

    rows = []
    for t in tables_sorted:
        rows.append([
            t.get("table_name", "?"),
            t.get("table_type", "?").upper(),
            str(len(t.get("columns", []))),
            t.get("description", "")[:70],
            ", ".join(t.get("source_tables", []))[:60],
        ])
    draw_table(["Table Name", "Type", "Cols", "Description", "Source Tables"],
               rows, [52, 22, 14, 90, 83])

    # ==========================================
    # RELATIONSHIPS
    # ==========================================
    pdf.add_page()
    section_title("03", "Relationship Specifications")
    body_text(f"{len(rels)} relationships define the foreign key connections between proposed tables.", 8)
    pdf.ln(2)

    if rels:
        rel_rows = [[r.get("from_table","?"), r.get("from_column","?"),
                      r.get("to_table","?"), r.get("to_column","?"),
                      r.get("cardinality","?"), r.get("description_english","")[:70]]
                     for r in rels]
        draw_table(["From Table", "From Column", "To Table", "To Column", "Card.", "Description"],
                   rel_rows, [46, 34, 46, 34, 16, 85])
    else:
        body_text("No relationships defined in this model.")

    # ==========================================
    # PER-TABLE SPECIFICATIONS
    # ==========================================
    for ti, t in enumerate(tables_sorted):
        pdf.add_page()
        tname = t.get("table_name", "unknown")
        ttype = t.get("table_type", "core_entity").lower()
        clr = CLR.get(ttype, BLUE)
        cols = t.get("columns", [])

        # Table header with accent color bar
        pdf.set_draw_color(*clr)
        pdf.set_line_width(1.5)
        pdf.line(M, M, M + 60, M)
        pdf.set_xy(M, M + 3)

        section_title(f"04.{ti+1}", f"{ttype.upper()}: {tname}")

        label_value("Description:", t.get("description", "N/A"))
        label_value("Source Tables:", ", ".join(t.get("source_tables", [])) or "N/A")
        label_value("Column Count:", str(len(cols)))
        pdf.ln(4)

        # Relationships for this table
        trels = [r for r in rels if r.get("from_table") == tname or r.get("to_table") == tname]
        if trels:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.cell(W, 5, f"Relationships ({len(trels)})", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            rel_rows = [[r.get("from_table",""), r.get("from_column",""), "->",
                          r.get("to_table",""), r.get("to_column",""),
                          r.get("cardinality",""), r.get("description_english","")[:60]]
                         for r in trels]
            draw_table(["From", "Column", "", "To", "Column", "Card.", "Description"],
                       rel_rows, [38, 28, 8, 38, 28, 14, 107])
            pdf.ln(3)

        # Column specifications
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(W, 5, "Column Specifications", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        rl = {"pk": "PK", "fk": "FK", "attribute": "Attr"}
        col_rows = [[c.get("name","?"), c.get("data_type","?"),
                      rl.get(c.get("role",""), c.get("role","")),
                      c.get("source","")[:45], c.get("description","")[:80]]
                     for c in cols]
        draw_table(["Column Name", "Data Type", "Role", "Source Mapping", "Description"],
                   col_rows, [48, 24, 18, 60, 111], accent=clr)

    # ==========================================
    # DROPPED COLUMNS
    # ==========================================
    if dropped:
        pdf.add_page()
        section_title("05", "Dropped Columns Analysis")
        body_text(f"{len(dropped)} columns were excluded from the 3NF relational model.", 8)
        pdf.ln(2)
        drop_rows = [[d.get("source_table","?"), d.get("column","?"), d.get("reason","")[:120]]
                      for d in dropped]
        draw_table(["Source Table", "Column", "Reason"],
                   drop_rows, [55, 45, 161])

    # ==========================================
    # APPENDIX
    # ==========================================
    pdf.add_page()
    section_title("A", "Appendix: Report Metadata")
    pdf.ln(2)
    label_value("Report Type:", "Relational Data Model (3NF) Design - Consulting Deliverable")
    label_value("Source Schema:", f"{catalog}.{schema}" if catalog else "N/A")
    label_value("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    label_value("Proposed Tables:", str(len(tables)))
    label_value("Total Columns:", str(total_cols))
    label_value("Relationships:", str(len(rels)))
    label_value("Dropped Columns:", str(len(dropped)))
    label_value("Table Types:", ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items())))
    pdf.ln(10)
    body_text("This report was auto-generated by the AI Data Model Designer. "
              "All specifications should be reviewed by a data architect before implementation.", 8)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# =============================================================================
# LLM INTEGRATION
# =============================================================================


# =============================================================================
# DASH APP LAYOUT
# =============================================================================

# Pre-load catalogs at startup using service principal (no user token needed)
_initial_catalogs = []
try:
    _sp = WorkspaceClient()
    _initial_catalogs = sorted(
        [{"label": c.name, "value": c.name} for c in _sp.catalogs.list() if c.name],
        key=lambda x: x["value"]
    )
    print(f"[app.py] Pre-loaded {len(_initial_catalogs)} catalogs at startup", flush=True)
except Exception as _e:
    print(f"[app.py] WARNING: Could not pre-load catalogs: {_e}", flush=True)

cyto.load_extra_layouts()
print("[app.py] Initializing Dash app...", flush=True)
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE
server = app.server  # For Databricks App hosting

# --- Debug endpoint (hit /debug to diagnose issues) ---
@server.route("/debug")
def debug_info():
    import json as _json
    info = {"request_headers": {}, "env": {}, "auth": {}, "catalogs": {}, "token_test": {}}
    
    # Show ALL request headers (key for finding the token header)
    for k, v in flask_request.headers:
        info["request_headers"][k] = v[:60] + "..." if len(v) > 60 else v
    
    for k in ["DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_WAREHOUSE_ID"]:
        v = os.environ.get(k, "NOT SET")
        info["env"][k] = v[:30] + "..." if len(v) > 30 else v
    info["env"]["CLIENT_SECRET_SET"] = bool(os.environ.get("DATABRICKS_CLIENT_SECRET"))
    info["env"]["TOKEN_SET"] = bool(os.environ.get("DATABRICKS_TOKEN"))
    
    # Test SP SDK
    try:
        _wc = WorkspaceClient()
        info["auth"]["sdk_auth_type"] = str(_wc.config.auth_type)
        cats = [c.name for c in _wc.catalogs.list()]
        info["auth"]["sp_catalog_count"] = len(cats)
    except Exception as e:
        info["auth"]["sp_error"] = str(e)[:200]
    
    # Test user token from various headers
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    
    for header_name in ["x-forwarded-access-token", "X-Forwarded-Access-Token", 
                         "x-databricks-token", "Authorization"]:
        token = flask_request.headers.get(header_name, "")
        if token:
            # Strip "Bearer " prefix if present
            if token.startswith("Bearer "):
                token = token[7:]
            try:
                _h = {"Authorization": f"Bearer {token}"}
                _r = requests.get(f"{host}/api/2.1/unity-catalog/catalogs", headers=_h, timeout=10)
                info["token_test"][header_name] = {
                    "found": True, "token_prefix": token[:15] + "...",
                    "uc_status": _r.status_code,
                    "catalog_count": len(_r.json().get("catalogs", [])) if _r.status_code == 200 else 0
                }
            except Exception as e:
                info["token_test"][header_name] = {"found": True, "error": str(e)[:100]}
        else:
            info["token_test"][header_name] = {"found": False}
    
    info["catalogs"]["pre_loaded_count"] = len(_initial_catalogs)
    
    # Test SQL warehouse with user token
    user_token = flask_request.headers.get("x-forwarded-access-token", "")
    if user_token:
        try:
            sql_rows = run_sql("SHOW CATALOGS", token=user_token)
            sql_cats = [r.get("catalog", r.get("catalog_name", "")) for r in sql_rows]
            info["sql_warehouse_test"] = {
                "status": "OK",
                "catalog_count": len(sql_cats),
                "first_10": sql_cats[:10]
            }
        except Exception as e:
            info["sql_warehouse_test"] = {"status": "FAILED", "error": str(e)[:300]}
    else:
        info["sql_warehouse_test"] = {"status": "NO_TOKEN"}
    
    return _json.dumps(info, indent=2, default=str), 200, {"Content-Type": "application/json"}

# --- API endpoints for catalog/schema loading (bypass Dash callbacks) ---
@server.route("/api/catalogs")
def api_catalogs():
    """Return catalog list as JSON — uses SQL warehouse (sees all 28 catalogs)."""
    try:
        token = flask_request.headers.get("x-forwarded-access-token", "")
        if token:
            rows = run_sql("SHOW CATALOGS", token=token)
            cats = sorted([r.get("catalog", r.get("catalog_name", "")) for r in rows
                          if r.get("catalog", r.get("catalog_name", ""))])
            print(f"[api] /api/catalogs -> {len(cats)} catalogs via SQL warehouse", flush=True)
            return {"catalogs": cats}, 200
    except Exception as e:
        print(f"[api] /api/catalogs SQL error: {e}", flush=True)
    # Fallback: UC REST API
    try:
        token = flask_request.headers.get("x-forwarded-access-token", "")
        if token:
            host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
            if host and not host.startswith("http"):
                host = f"https://{host}"
            resp = requests.get(f"{host}/api/2.1/unity-catalog/catalogs",
                                headers={"Authorization": f"Bearer {token}"}, timeout=15)
            if resp.status_code == 200:
                cats = sorted([c.get("name", "") for c in resp.json().get("catalogs", []) if c.get("name")])
                return {"catalogs": cats}, 200
    except Exception as e:
        print(f"[api] /api/catalogs REST error: {e}", flush=True)
    return {"catalogs": [c["value"] for c in _initial_catalogs]}, 200


@server.route("/api/schemas/<catalog>")
def api_schemas(catalog):
    """Return schema list for a catalog as JSON."""
    try:
        token = flask_request.headers.get("x-forwarded-access-token", "")
        if token:
            rows = run_sql(f"SHOW SCHEMAS IN `{catalog}`", token=token)
            schemas = sorted([r.get("databaseName", r.get("namespace", r.get("schema_name", "")))
                             for r in rows
                             if r.get("databaseName", r.get("namespace", r.get("schema_name", "")))
                             and r.get("databaseName", r.get("namespace", r.get("schema_name", ""))) != "information_schema"])
            print(f"[api] /api/schemas/{catalog} -> {len(schemas)} schemas", flush=True)
            return {"schemas": schemas}, 200
    except Exception as e:
        print(f"[api] /api/schemas/{catalog} error: {e}", flush=True)
    return {"schemas": []}, 200



app.layout = html.Div(
    [
        # Hidden stores
        dcc.Store(id="model-store", data=None),
        dcc.Download(id="download-pdf"),
        dcc.Store(id="current-catalog", data=""),
        dcc.Store(id="current-schema", data=""),
        dcc.Interval(id="init-interval", interval=2000, max_intervals=1, n_intervals=0),
        dcc.Interval(id="poll-interval", interval=POLL_INTERVAL_MS, disabled=True),

        # Controls bar (labels above dropdowns)
        html.Div(
            [
                html.Div([
                    html.Label("CATALOG", className="control-label"),
                    dcc.Dropdown(id="catalog-dd", options=[], value=None,
                                 placeholder="Select catalog...",
                                 className="dash-dropdown", style={"width": "300px"}),
                ], className="control-group"),
                html.Div([
                    html.Label("SCHEMA", className="control-label"),
                    dcc.Dropdown(id="schema-dd", options=[], value=None,
                                 placeholder="Select schema...",
                                 className="dash-dropdown", style={"width": "400px"}),
                ], className="control-group"),
                html.Div([
                    html.Label("MODEL TYPE", className="control-label"),
                    dcc.RadioItems(
                        id="model-type-radio",
                        options=[
                            {"label": " 3NF Relational", "value": "3nf"},
                            {"label": " Dimensional (Star)", "value": "dimensional"},
                        ],
                        value="3nf",
                        inline=True,
                        className="model-type-radio",
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px", "fontSize": "13px", "fontWeight": "500",
                                     "color": "#334155", "cursor": "pointer"},
                    ),
                ], className="control-group"),
                html.Div(style={"flex": "1"}),
                html.Button(
                    "\U0001f916 Generate Data Model",
                    id="generate-btn",
                    className="generate-btn",
                ),
            ],
            className="controls-bar",
        ),

        # Graph area with optional right detail panel
        html.Div(
            [
                # Graph
                html.Div(
                    [
                        cyto.Cytoscape(
                            id="erd-graph",
                            elements=[],
                            stylesheet=get_cytoscape_stylesheet(),
                            layout={"name": "fcose", "animate": False, "randomize": True, "quality": "proof", "nodeRepulsion": 15000, "idealEdgeLength": 250, "edgeElasticity": 0.35, "componentSpacing": 400, "nestingFactor": 0.1, "gravity": 0.08, "gravityRange": 5.0, "fit": True, "padding": 80},
                            style={"width": "100%", "height": "calc(100vh - 160px)",
                                   "backgroundColor": "#f8fafc"},
                        ),
                        html.Div(
                            [
                                html.Button("+", id="zoom-in", className="zoom-btn"),
                                html.Button("\u2013", id="zoom-out", className="zoom-btn"),
                                html.Button("\u2b1c", id="zoom-fit", className="zoom-btn"),
                            ],
                            className="zoom-controls",
                        ),
                        dcc.Input(
                            id="search-input", type="text", placeholder="Search tables...",
                            className="search-input",
                            debounce=True,
                            style={"position": "absolute", "top": "10px", "left": "10px", "zIndex": "100"},
                        ),
                    ],
                    style={"flex": "1", "position": "relative", "minWidth": "0"},
                ),
                # Right detail panel (hidden until node clicked)
                html.Div(
                    id="detail-panel",
                    children=[
                        html.Div([
                            html.Div("\U0001f4cb", style={"fontSize": "36px", "textAlign": "center", "marginBottom": "8px"}),
                            html.P("Select a table node", style={"fontWeight": "600", "textAlign": "center", "margin": "0"}),
                            html.P("Click any node in the graph to inspect its columns and relationships.",
                                   style={"color": "#94a3b8", "textAlign": "center", "fontSize": "13px"}),
                        ], style={"padding": "40px 20px"}),
                    ],
                    style={"width": "320px", "borderLeft": "1px solid #e2e8f0",
                           "overflowY": "auto", "height": "calc(100vh - 160px)", "background": "#ffffff"},
                ),
            ],
            style={"display": "flex", "width": "100%"},
        ),

        # Legend bar — dynamic based on model type
        html.Div(
            [
                html.Div(
                    id="legend-bar",
                    children=[
                        html.Span("\u25cf", style={"color": COLORS["core_entity"], "marginRight": "4px"}),
                        html.Span("Core Entity", style={"marginRight": "16px"}),
                        html.Span("\u25cf", style={"color": COLORS["weak_entity"], "marginRight": "4px"}),
                        html.Span("Weak Entity", style={"marginRight": "16px"}),
                        html.Span("\u25cf", style={"color": COLORS["associative"], "marginRight": "4px"}),
                        html.Span("Associative", style={"marginRight": "16px"}),
                        html.Span("\u25cf", style={"color": COLORS["reference"], "marginRight": "4px"}),
                        html.Span("Reference", style={"marginRight": "16px"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "flex": "1"},
                ),
                html.Button("Export Report", id="export-btn", className="export-btn", disabled=True),
            ],
            className="legend-bar",
        ),

        # Summary section (full width, below graph)
        html.Div(id="summary-panel", style={"width": "100%"}),

        # Thinking modal overlay
        html.Div(
            id="thinking-overlay",
            children=[
                html.Div(
                    [
                        html.Div([
                            html.Span("\U0001f9e0", style={"fontSize": "32px", "marginRight": "12px"}),
                            html.Div([
                                html.H3("Designing your data model...",
                                         style={"margin": "0", "fontSize": "18px", "fontWeight": "600"}),
                                html.Span(id="elapsed-timer", children="0s elapsed",
                                          style={"color": "#94a3b8", "fontSize": "13px"}),
                            ]),
                            html.Div(style={"flex": "1"}),
                            html.Button("\u2715 Stop", id="cancel-btn", className="kill-btn"),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
                        html.Div(id="thinking-steps", style={"maxHeight": "400px", "overflowY": "auto"}),
                    ],
                    className="thinking-modal",
                )
            ],
            className="thinking-overlay",
            style={"display": "none"},
        ),


        # Powered by Databricks footer
        html.Div(
            [
                html.Span("Powered by"),
                html.Img(src="/assets/db-log.png", style={"height": "22px"}),
            ],
            className="powered-footer",
        ),

        # Auto-fit script
        html.Script("""
        (function() {
            function doFit() {
                try {
                    var el = document.getElementById('erd-graph');
                    if (el && el._cyreg && el._cyreg.cy) {
                        var cy = el._cyreg.cy;
                        if (cy.nodes().length > 0) {
                            cy.fit(undefined, 40);
                            cy.center();
                        }
                    }
                } catch(e) {}
            }
            var observer = new MutationObserver(function() {
                setTimeout(doFit, 500);
                setTimeout(doFit, 1500);
                setTimeout(doFit, 3000);
                setTimeout(doFit, 5000);
            });
            var target = document.getElementById('erd-graph');
            if (target) {
                observer.observe(target, {childList: true, subtree: true});
            }
        })();
        """),
    ],
    style={"backgroundColor": "#ffffff", "minHeight": "100vh", "fontFamily": "'Inter', sans-serif", "color": "#1e293b"},
)



# =============================================================================
# CALLBACKS
# =============================================================================

# --- Load catalogs on page load (triggers on component mount) ---
@app.callback(
    Output("catalog-dd", "options"),
    Output("catalog-dd", "placeholder"),
    Input("catalog-dd", "id"),
)
def load_catalogs(_):
    """Auto-fires on page load via Input('catalog-dd', 'id')."""
    try:
        rows = run_sql("SHOW CATALOGS")
        names = sorted(r.get("catalog") or list(r.values())[0] for r in rows)
        opts = [{"label": n, "value": n} for n in names]
        print(f"[app.py] load_catalogs: {len(opts)} catalogs", flush=True)
        return opts, "Select catalog..."
    except Exception as e:
        print(f"[app.py] load_catalogs error: {e}", flush=True)
        return [], str(e)[:80]


# --- Load schemas when catalog is selected ---
@app.callback(
    Output("schema-dd", "options"),
    Output("schema-dd", "value"),
    Input("catalog-dd", "value"),
)
def load_schemas(catalog):
    if not catalog:
        return [], None
    try:
        rows = run_sql(f"SHOW SCHEMAS IN `{catalog}`")
        names = sorted(
            r.get("databaseName") or r.get("namespace") or r.get("schema_name") or list(r.values())[0]
            for r in rows
        )
        names = [n for n in names if n and n != "information_schema"]
        opts = [{"label": n, "value": n} for n in names]
        print(f"[app.py] load_schemas: {len(opts)} schemas for {catalog}", flush=True)
        return opts, None
    except Exception as e:
        print(f"[app.py] load_schemas error: {e}", flush=True)
        return [], None
# --- Generate Data Model ---
@app.callback(
    Output("thinking-overlay", "style"),
    Output("poll-interval", "disabled"),
    Input("generate-btn", "n_clicks"),
    State("catalog-dd", "value"),
    State("schema-dd", "value"),
    State("model-type-radio", "value"),
    prevent_initial_call=True,
)
def on_generate(n_clicks, catalog, schema, model_type):
    if not catalog or not schema:
        return {"display": "none"}, True

    token = _get_token()
    _reset_job()

    _job["_catalog"] = catalog
    _job["_schema"] = schema
    _job["_model_type"] = model_type or "3nf"
    thread = threading.Thread(
        target=_bg_generate,
        args=(catalog, schema, model_type or "3nf", token),
        daemon=True,
    )
    thread.start()

    return {"display": "flex"}, False  # Show modal, enable polling


# --- Kill switch ---
@app.callback(
    Output("cancel-btn", "n_clicks"),
    Input("cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_cancel(n_clicks):
    _job["cancelled"] = True
    return no_update


# --- Poll progress ---
@app.callback(
    Output("thinking-steps", "children"),
    Output("elapsed-timer", "children"),
    Output("erd-graph", "elements"),
    Output("erd-graph", "layout"),
    Output("summary-panel", "children"),
    Output("model-store", "data"),
    Output("export-btn", "disabled"),
    Output("thinking-overlay", "style", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_progress(n):
    # Elapsed timer
    elapsed = ""
    if _job.get("start_time"):
        secs = int((datetime.now() - _job["start_time"]).total_seconds()) if _job.get("start_time") else 0
        elapsed = f"{secs}s elapsed"

    # Build steps display with checkmarks
    step_items = []
    for s in _job.get("steps", []):
        icon = {"active": "\u23f3", "done": "\u2705", "error": "\u274c"}.get(s["status"], "\u2022")
        cls = f"step-item step-{s['status']}"
        step_items.append(
            html.Div(
                [
                    html.Span(icon, className="step-icon"),
                    html.Span(s["time"], className="step-time"),
                    html.Span(s["msg"], className="step-msg"),
                ],
                className=cls,
            )
        )

    # Check if complete
    if not _job.get("active", True):
        if _job.get("result"):
            model = _job["result"]
            # Add catalog/schema context for PDF report
            model["_catalog"] = _job.get("_catalog", "")
            model["_schema"] = _job.get("_schema", "")
            model["_model_type"] = _job.get("_model_type", "3nf")
            elements = build_proposed_erd_elements(model)
            summary = build_summary_panel(model)
            # Dynamic layout: scale repulsion & spacing based on entity count
            _n = len(elements)
            _repulsion = max(15000, _n * 200)   # More entities → more repulsion
            _spacing = max(400, _n * 5)          # More entities → more inter-cluster spacing
            _edge_len = max(250, _n * 3)         # More entities → longer edges
            _cola = {"name": "fcose", "animate": False, "randomize": True, "quality": "proof", "nodeRepulsion": _repulsion, "idealEdgeLength": _edge_len, "edgeElasticity": 0.35, "componentSpacing": _spacing, "nestingFactor": 0.1, "gravity": 0.08, "gravityRange": 5.0, "fit": True, "padding": 80}
            return step_items, elapsed, elements, _cola, summary, model, False, {"display": "none"}, True
        elif _job.get("error"):
            err_msg = _job["error"]
            error_panel = html.Div(
                [html.H4("Error"), html.P(err_msg, style={"color": "#dc2626"})],
                className="summary-card",
            )
            return step_items, elapsed, [], no_update, error_panel, None, True, {"display": "none"}, True

    # Still in progress
    return step_items, elapsed, no_update, no_update, no_update, no_update, no_update, no_update, no_update



# --- Search filter ---
@app.callback(
    Output("erd-graph", "stylesheet"),
    Input("search-input", "value"),
)
def on_search(query):
    base = get_cytoscape_stylesheet()
    if not query or not query.strip():
        return base

# --- Legend updater (model type aware) ---
@app.callback(
    Output("legend-bar", "children"),
    Input("model-store", "data"),
    prevent_initial_call=True,
)
def update_legend(model):
    mt = (model or {}).get("_model_type", "3nf") if model else "3nf"
    dot = lambda c: html.Span("\u25cf", style={"color": c, "marginRight": "4px"})
    lbl = lambda t: html.Span(t, style={"marginRight": "16px"})
    if mt == "dimensional":
        items = [
            dot(COLORS["fact"]), lbl("Fact"),
            dot(COLORS["dimension"]), lbl("Dimension"),
            dot(COLORS["bridge"]), lbl("Bridge"),
            dot(COLORS["aggregate"]), lbl("Aggregate"),
        ]
    else:
        items = [
            dot(COLORS["core_entity"]), lbl("Core Entity"),
            dot(COLORS["weak_entity"]), lbl("Weak Entity"),
            dot(COLORS["associative"]), lbl("Associative"),
            dot(COLORS["reference"]), lbl("Reference"),
        ]
    return items


    q = query.strip().lower()
    # Add dimmed class to non-matching nodes
    base.append({
        "selector": f"node[label !*= '{q}']",  # Does not contain
        "style": {"opacity": 0.15},
    })
    base.append({
        "selector": f"node[label *= '{q}']",  # Contains
        "style": {"border-width": 4, "border-color": "#facc15"},
    })
    return base


# --- Node click -> detail in summary ---
@app.callback(
    Output("detail-panel", "children"),
    Input("erd-graph", "tapNodeData"),
    State("model-store", "data"),
    prevent_initial_call=True,
)
def on_node_click(node_data, model):
    if not node_data or not model:
        return no_update

    tname = node_data.get("id", "")

    # Find table in model
    table_info = None
    for t in model.get("proposed_tables", []):
        if t.get("table_name") == tname:
            table_info = t
            break

    if not table_info:
        return no_update

    ttype = table_info.get("table_type", "core_entity")
    cols = table_info.get("columns", [])
    pks = sum(1 for c in cols if c.get("role", "").lower() == "pk")

    # Find relationships for this table
    rels = [r for r in model.get("relationships", [])
            if r.get("from_table") == tname or r.get("to_table") == tname]

    children = [
        # Type badge
        html.Span(ttype.upper(), className=f"badge badge-{ttype}"),
        # Table name
        html.H3(tname, style={"margin": "8px 0 4px", "fontSize": "16px", "fontWeight": "700"}),
        # Description
        html.P(table_info.get("description", ""),
               style={"color": "#64748b", "fontSize": "13px", "lineHeight": "1.5", "margin": "0 0 12px"}),
        # Stats badges
        html.Div([
            html.Span(f"{len(cols)} cols", style={
                "background": "#dcfce7", "color": "#15803d", "padding": "3px 10px",
                "borderRadius": "12px", "fontSize": "12px", "fontWeight": "600", "marginRight": "6px"}),
            html.Span(f"{pks} PK", style={
                "background": "#fef3c7", "color": "#92400e", "padding": "3px 10px",
                "borderRadius": "12px", "fontSize": "12px", "fontWeight": "600"}),
        ], style={"marginBottom": "16px"}),

        # COLUMNS section
        html.H4("COLUMNS", style={"fontSize": "11px", "fontWeight": "700", "color": "#64748b",
                                    "letterSpacing": "0.5px", "borderBottom": "1px solid #e2e8f0",
                                    "paddingBottom": "6px", "marginBottom": "8px"}),
    ]

    for c in cols:
        role = c.get("role", "attribute")
        icon = {"pk": "\U0001f511", "fk": "\U0001f517", "measure": "\U0001f4ca", "attribute": ""}.get(role, "")
        children.append(
            html.Div([
                html.Span(f"{icon} " if icon else "", style={"fontSize": "12px"}),
                html.Span(c.get("name", ""), style={"fontWeight": "500", "fontSize": "13px"}),
                html.Span(c.get("data_type", "").upper(),
                          style={"float": "right", "color": "#7c3aed", "fontSize": "11px", "fontWeight": "600"}),
            ], style={"padding": "6px 0", "borderBottom": "1px solid #f1f5f9"})
        )

    # RELATIONSHIPS section
    if rels:
        children.append(html.H4(f"RELATIONSHIPS ({len(rels)})", style={
            "fontSize": "11px", "fontWeight": "700", "color": "#64748b",
            "letterSpacing": "0.5px", "borderBottom": "1px solid #e2e8f0",
            "paddingBottom": "6px", "marginTop": "16px", "marginBottom": "8px"}))
        for r in rels:
            other = r.get("to_table") if r.get("from_table") == tname else r.get("from_table")
            card = r.get("cardinality", "")
            desc = r.get("description_english", "")
            children.append(
                html.Div([
                    html.Span(other, style={"fontWeight": "600", "fontSize": "13px"}),
                    html.Span(f" ({card})", style={"color": "#94a3b8", "fontSize": "11px"}) if card else "",
                    html.Br(),
                    html.Span(desc, style={"color": "#64748b", "fontSize": "12px", "fontStyle": "italic"}),
                ], style={"padding": "6px 0", "borderBottom": "1px solid #f1f5f9"})
            )

    return html.Div(children, style={"padding": "16px"})


# --- Back to overview ---
@app.callback(
    Output("detail-panel", "children", allow_duplicate=True),
    Input("back-to-overview", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_overview(n_clicks):
    return html.Div([
        html.Div("\U0001f4cb", style={"fontSize": "36px", "textAlign": "center", "marginBottom": "8px"}),
        html.P("Select a table node", style={"fontWeight": "600", "textAlign": "center", "margin": "0"}),
        html.P("Click any node in the graph to inspect its columns and relationships.",
               style={"color": "#94a3b8", "textAlign": "center", "fontSize": "13px"}),
    ], style={"padding": "40px 20px"})





# --- Zoom controls (clientside) ---
app.clientside_callback(
    """
    function(n) {
        try {
            var el = document.getElementById('erd-graph');
            if (el && el._cyreg) el._cyreg.cy.zoom(el._cyreg.cy.zoom() * 1.2);
        } catch(e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output("zoom-in", "n_clicks"),
    Input("zoom-in", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n) {
        try {
            var el = document.getElementById('erd-graph');
            if (el && el._cyreg) el._cyreg.cy.zoom(el._cyreg.cy.zoom() * 0.8);
        } catch(e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output("zoom-out", "n_clicks"),
    Input("zoom-out", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n) {
        try {
            var el = document.getElementById('erd-graph');
            if (el && el._cyreg) { el._cyreg.cy.fit(undefined, 40); el._cyreg.cy.center(); }
        } catch(e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output("zoom-fit", "n_clicks"),
    Input("zoom-fit", "n_clicks"),
    prevent_initial_call=True,
)



# --- Export PDF (server-side generation) ---
@app.callback(
    Output("download-pdf", "data"),
    Input("export-btn", "n_clicks"),
    State("model-store", "data"),
    prevent_initial_call=True,
)
def export_pdf_report(n_clicks, model):
    if not n_clicks or not model:
        return no_update
    try:
        pdf_bytes = _generate_consulting_report(model)
        schema_name = model.get("_schema", "data_model")
        fname = f"Data_Model_Report_{schema_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes, filename=fname)
    except Exception as e:
        print(f"[PDF] Export error: {e}", flush=True)
        traceback.print_exc()
        return no_update

print(f"[app.py] All callbacks registered. Starting server...", flush=True)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
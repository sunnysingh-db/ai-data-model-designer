# =============================================================================
# AI Data Model Designer — Main Application
# =============================================================================
# A Databricks App (Dash/Cytoscape) that analyzes Unity Catalog schemas
# and proposes optimized star/snowflake dimensional models using LLM.
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
LLM_WORKERS = 12
LLM_GROUP_SIZE = 5
API_TIMEOUT = 300
MAX_TOKENS_SINGLE = 32000
MAX_TOKENS_GROUP = 16000
MAX_PAYLOAD_CHARS = 500000
POLL_INTERVAL_MS = 1200

# Color scheme
COLORS = {
    "fact": "#2563eb",
    "dimension": "#16a34a",
    "bridge": "#ea580c",
    "aggregate": "#7c3aed",
    "pk": "#eab308",
    "fk": "#2563eb",
    "measure": "#16a34a",
    "attribute": "#94a3b8",
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
            # Negate for descending sort (higher version first)
            version = tuple(-int(n) for n in nums) if nums else (0,)
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
    return [dict(zip(cols, row)) for row in resp.result.data_array]


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
        print(f"[LLM] → {_short_model(model)} | prompt={len(prompt):,} chars | max_tokens={max_tokens}", flush=True)

        for attempt in range(2):  # Max 1 retry per model
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    content_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    elapsed = time.time() - _t0
                    print(f"[LLM] ← {_short_model(model)} | {len(content_text):,} chars | {elapsed:.1f}s", flush=True)
                    return content_text
                elif resp.status_code == 429:
                    print(f"[LLM] ⚠ {_short_model(model)} → 429 rate limited (attempt {attempt+1}/2)", flush=True)
                    if attempt == 0:
                        time.sleep(15)
                        continue
                    last_error = f"429 rate limited on {model}"
                    break
                elif resp.status_code in (500, 502, 503):
                    print(f"[LLM] ⚠ {_short_model(model)} → {resp.status_code} service error, trying next...", flush=True)
                    last_error = f"{resp.status_code} on {model}"
                    break  # Fall to next model
                elif resp.status_code in (400, 401, 403):
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

    # Select models: explicit list, or top 3 Opus by default
    if use_models:
        race_models = use_models
    else:
        opus_models = [m for m in LLM_FALLBACK_CHAIN if "opus" in m.lower()]
        race_models = opus_models[:2] if opus_models else LLM_FALLBACK_CHAIN[:2]

    _short = lambda m: m.replace("databricks-claude-", "")
    ep_names = [_short(m) for m in race_models]
    print(f"[RACE] 🏁 Racing {len(race_models)} Opus endpoints: {', '.join(ep_names)} | prompt={len(prompt):,} chars", flush=True)
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
    return _build_profiling_payload(profiles, catalog, schema)


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

def _build_system_prompt():
    return (
        "You are a senior data architect with 20 years of experience in dimensional modeling, "
        "star schemas, and data warehouse design. You are precise, thorough, and always produce valid JSON."
    )


def _build_analysis_prompt(payload, catalog, schema):
    return f"""{_build_system_prompt()}

Analyze the following schema from `{catalog}`.`{schema}` and design an OPTIMAL dimensional model.

RULES:
1. Do NOT replicate existing tables. REDESIGN them into a proper star/snowflake model.
2. Use dim_ prefix for dimensions, fact_ for facts, bridge_ for bridges, agg_ for aggregates.
3. Every FK must reference a valid PK in another proposed table.
4. Be CONCISE in descriptions (15 words max each).
5. Use action-verb relationship descriptions.
6. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP.
7. ALWAYS include a dim_date conformed dimension if ANY source table has DATE or TIMESTAMP columns.
   dim_date MUST include at minimum these columns (all source="generated"):
   - date_key (LONG, PK, YYYYMMDD surrogate key)
   - full_date (DATE)
   - year (LONG), quarter (STRING, e.g. "Q1"), month (LONG), month_name (STRING)
   - week_of_year (LONG), day (LONG), day_of_week (STRING), day_of_year (LONG)
   - is_weekend (BOOLEAN), is_month_end (BOOLEAN)
   Link every DATE/TIMESTAMP FK in fact tables to dim_date.date_key.

PROFILING DATA:
{payload}

Return ONLY valid JSON with this exact structure:
{{
  "proposed_tables": [
    {{
      "table_name": "dim_xxx",
      "table_type": "fact|dimension|bridge|aggregate",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {{
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|fk|measure|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "from_table": "fact_xxx",
      "from_column": "key_col",
      "to_table": "dim_xxx",
      "to_column": "key_col",
      "cardinality": "1:N",
      "description_english": "Each X has many Y"
    }}
  ],
  "columns_dropped": [
    {{
      "source_table": "table",
      "column": "col",
      "reason": "why dropped"
    }}
  ],
  "data_summary": "3-4 sentences summarizing the data domain and model rationale.",
  "recommended_consumers": ["Analytics Team", "Finance"]
}}

Return ONLY the JSON. No markdown fences. No extra text."""


def _build_group_prompt(payload, catalog, schema, all_table_names):
    all_names = ", ".join(all_table_names)
    return f"""{_build_system_prompt()}

You are analyzing a SUBSET of tables from `{catalog}`.`{schema}`.
All tables in the schema: {all_names}

Analyze the following tables and propose dimensional model components.
Note cross-references to tables NOT in this subset — they will be merged later.

RULES:
1. Do NOT replicate existing tables. REDESIGN them into a proper star/snowflake model.
2. Use dim_ prefix for dimensions, fact_ for facts, bridge_ for bridges, agg_ for aggregates.
3. Every FK must reference a valid PK in another proposed table.
4. Be CONCISE in descriptions (15 words max each).
5. Normalize data types to: STRING, LONG, DOUBLE, DATE, BOOLEAN, TIMESTAMP.
6. ALWAYS include a dim_date conformed dimension if ANY source table has DATE or TIMESTAMP columns.
   dim_date MUST include at minimum these columns (all source="generated"):
   - date_key (LONG, PK, YYYYMMDD surrogate key)
   - full_date (DATE)
   - year (LONG), quarter (STRING, e.g. "Q1"), month (LONG), month_name (STRING)
   - week_of_year (LONG), day (LONG), day_of_week (STRING), day_of_year (LONG)
   - is_weekend (BOOLEAN), is_month_end (BOOLEAN)
   Link every DATE/TIMESTAMP FK in fact tables to dim_date.date_key.

PROFILING DATA:
{payload}

Return ONLY valid JSON with this EXACT structure (use these EXACT key names):
{{
  "proposed_tables": [
    {{
      "table_name": "dim_xxx",
      "table_type": "fact|dimension|bridge|aggregate",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {{
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|fk|measure|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "from_table": "fact_xxx",
      "from_column": "key_col",
      "to_table": "dim_xxx",
      "to_column": "key_col",
      "cardinality": "1:N",
      "description_english": "Each X has many Y"
    }}
  ],
  "columns_dropped": [
    {{
      "source_table": "table",
      "column": "col",
      "reason": "why dropped"
    }}
  ],
  "data_summary": "3-4 sentences summarizing the data domain and model rationale.",
  "recommended_consumers": ["Analytics Team", "Finance"]
}}
"""


def _build_merge_prompt(group_results, catalog, schema):
    combined = json.dumps(group_results, indent=2)
    return f"""{_build_system_prompt()}

You have received multiple partial dimensional model proposals for `{catalog}`.`{schema}`.
Merge them into a single, unified model.

RULES:
1. Deduplicate tables that appear in multiple groups.
2. Add cross-group relationships where FKs reference tables from other groups.
3. Ensure every FK references a valid PK in a proposed table.
4. Write a unified data_summary (3-4 sentences).
5. Combine all recommended_consumers (deduplicated).
6. Combine all columns_dropped.

GROUP PROPOSALS:
{combined}

Return ONLY valid JSON with this EXACT structure (use these EXACT key names — do NOT rename fields):
{{
  "proposed_tables": [
    {{
      "table_name": "dim_xxx",
      "table_type": "fact|dimension|bridge|aggregate",
      "description": "Short description",
      "source_tables": ["original_table"],
      "columns": [
        {{
          "name": "col_name",
          "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN|TIMESTAMP",
          "role": "pk|fk|measure|attribute",
          "source": "original_table.column or generated",
          "description": "Short description"
        }}
      ]
    }}
  ],
  "relationships": [
    {{
      "from_table": "fact_xxx",
      "from_column": "key_col",
      "to_table": "dim_xxx",
      "to_column": "key_col",
      "cardinality": "1:N",
      "description_english": "Each X has many Y"
    }}
  ],
  "columns_dropped": [
    {{
      "source_table": "table",
      "column": "col",
      "reason": "why dropped"
    }}
  ],
  "data_summary": "3-4 sentences summarizing the data domain and model rationale.",
  "recommended_consumers": ["Analytics Team", "Finance"]
}}
"""


# =============================================================================
# LLM ANALYSIS PIPELINE
# =============================================================================

def _analyze_single_pass(payload, catalog, schema, token):
    """For small schemas: single Opus call with fallback chain. No racing = zero waste."""
    opus_models = [m for m in LLM_FALLBACK_CHAIN if "opus" in m.lower()]
    model_chain = opus_models[:2] if opus_models else LLM_FALLBACK_CHAIN[:2]
    _short = lambda m: m.replace("databricks-claude-", "")
    ep_names = [_short(m) for m in model_chain]
    prompt = _build_analysis_prompt(payload, catalog, schema)
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


def _analyze_groups(payload_per_table, catalog, schema, token, all_table_names, group_size=None):
    """Distribute table groups across Opus endpoints in parallel.

    group_size: dynamic — calculated to maximize parallelism across endpoints.
    Each group assigned to a specific Opus endpoint (round-robin).
    Every API call does productive work — zero wasted compute.
    """
    _short = lambda m: m.replace("databricks-claude-", "")
    opus_models = [m for m in LLM_FALLBACK_CHAIN if "opus" in m.lower()]
    work_models = opus_models[:2] if opus_models else LLM_FALLBACK_CHAIN[:2]
    n_eps = len(work_models)

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
        # Fallback to other Opus if primary fails
        fallback = [m for m in work_models if m != ep] + [m for m in LLM_FALLBACK_CHAIN if m not in work_models]
        _g_t0 = time.time()
        group_payload = "\n".join(payload_per_table[t] for t in group_tables if t in payload_per_table)
        prompt = _build_group_prompt(group_payload, catalog, schema, all_table_names)
        print(f"[GROUP] ▶ Group {endpoint_idx+1} started → {_short(ep)} | {len(group_tables)} tables | prompt={len(prompt):,} chars", flush=True)
        result = _call_chat_model(prompt, token, MAX_TOKENS_GROUP, model_chain=[ep] + fallback)
        elapsed = time.time() - _g_t0
        print(f"[GROUP] ✓ Group {endpoint_idx+1} done → {_short(ep)} | {len(result):,} chars | {elapsed:.1f}s", flush=True)
        return result, endpoint_idx

    _add_step(f"⏳ {len(groups)} Opus calls running in parallel...")

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
                    parsed = _parse_llm_json(result_text)
                    if parsed:
                        n_tables = len(parsed.get("proposed_tables", []))
                        group_results.append(parsed)
                        print(f"[GROUP] ✓ Group {gidx+1} parsed: {n_tables} tables proposed", flush=True)
            except Exception as e:
                ep = LLM_FALLBACK_CHAIN[idx % n_eps] if LLM_FALLBACK_CHAIN else "?"
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
    merged_consumers = set()

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

        # Collect consumers
        for c in g.get("recommended_consumers", []):
            merged_consumers.add(c)

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
        "recommended_consumers": sorted(merged_consumers),
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

def _parse_llm_json(raw):
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

    # Normalize table types
    valid_types = {"fact", "dimension", "bridge", "aggregate"}
    for t in data.get("proposed_tables", []):
        tt = t.get("table_type", "").lower()
        if tt not in valid_types:
            if "fact" in t.get("table_name", "").lower():
                t["table_type"] = "fact"
            elif "dim" in t.get("table_name", "").lower():
                t["table_type"] = "dimension"
            elif "bridge" in t.get("table_name", "").lower():
                t["table_type"] = "bridge"
            elif "agg" in t.get("table_name", "").lower():
                t["table_type"] = "aggregate"
            else:
                t["table_type"] = "dimension"
        else:
            t["table_type"] = tt

        # Normalize column roles
        valid_roles = {"pk", "fk", "measure", "attribute"}
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
    data.setdefault("recommended_consumers", [])

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

def _bg_generate(catalog, schema, layout, token):
    """Background thread: profile schema, call LLM, parse result."""
    _tls.token = token
    try:
        # Step 1: Profile
        payload = _profile_schema(catalog, schema, token)
        if not payload:
            _job["error"] = "No tables found in schema"
            _job["active"] = False
            return

        _check_cancelled()

        # Step 2: LLM Analysis — ALWAYS distribute across Opus endpoints
        table_count = payload.count("\nTable: ")
        opus_models = [m for m in LLM_FALLBACK_CHAIN if "opus" in m.lower()]
        work_models = opus_models[:2] if opus_models else LLM_FALLBACK_CHAIN[:2]
        ep_names = [m.replace("databricks-claude-", "") for m in work_models]
        n_eps = len(work_models)
        print(f"[GEN] LLM phase: {table_count} tables → distributing across {n_eps} Opus endpoints, payload={len(payload):,} chars", flush=True)

        # Build per-table payloads
        sections = re.split(r"(?=\nTable: )", payload)
        header = sections[0] if sections else ""
        per_table = {}
        for sec in sections[1:]:
            match = re.match(r"\nTable: (\S+)", sec)
            if match:
                per_table[match.group(1)] = header + sec

        # Sanity check: did we parse all tables?
        if len(per_table) < table_count:
            print(f"[WARN] Parsed {len(per_table)} tables from payload, but profiled {table_count}. Possible truncation.", flush=True)
            _add_step(f"⚠ Parsed {len(per_table)}/{table_count} tables (payload may have been truncated)", status="error")

        if len(per_table) == 1:
            # Edge case: only 1 table — direct call, no distribution needed
            _add_step(f"🧠 Analyzing 1 table via {ep_names[0]} (Opus)")
            raw_response = _analyze_single_pass(payload, catalog, schema, token)
        else:
            # 1 table per API call = maximum parallelism
            # Round-robin across Opus endpoints (opus-4-7, opus-4-6, opus-4-7, ...)
            _add_step(f"⚡ {len(per_table)} parallel LLM calls (1 table each) → round-robin {', '.join(ep_names)}")
            raw_response = _analyze_groups(per_table, catalog, schema, token, list(per_table.keys()), 1)

        _check_cancelled()

        # Step 3: Parse
        _add_step(f"🔍 Parsing LLM response ({len(raw_response):,} chars) — extracting JSON model...")
        model = _parse_llm_json(raw_response)
        if not model:
            _job["error"] = "Failed to parse LLM response as valid JSON"
            _add_step("❌ Parse failed — LLM returned invalid JSON", status="error")
            print(f"[PARSE] Failed. First 500 chars: {raw_response[:500]}", flush=True)
        else:
            t_count = len(model.get("proposed_tables", []))
            r_count = len(model.get("relationships", []))
            col_count = sum(len(t.get("columns", [])) for t in model.get("proposed_tables", []))
            _add_step(f"✅ Model ready — {t_count} tables, {r_count} relationships, {col_count} columns", status="done")
            _job["result"] = model
            print(f"[PARSE] ✅ {t_count} tables, {r_count} rels, {col_count} cols", flush=True)

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

def build_proposed_erd_elements(model):
    """Build Cytoscape elements from LLM model proposal."""
    elements = []
    if not model:
        return elements

    # Nodes — one per proposed table
    for t in model.get("proposed_tables", []):
        tname = t.get("table_name", "unknown")
        ttype = t.get("table_type", "dimension").lower()
        col_count = len(t.get("columns", []))
        color = COLORS.get(ttype, COLORS["attribute"])

        elements.append({
            "data": {
                "id": tname,
                "label": f"{tname}\n({col_count} cols)",
                "table_type": ttype,
                "color": color,
                "description": t.get("description", ""),
                "columns": json.dumps(t.get("columns", [])),
                "source_tables": json.dumps(t.get("source_tables", [])),
            },
            "classes": ttype,
        })

    # Edges — consolidated: ONE per table pair
    edge_map = defaultdict(list)  # (from_table, to_table) -> [relationships]
    for r in model.get("relationships", []):
        ft = r.get("from_table", "")
        tt = r.get("to_table", "")
        if ft and tt:
            key = tuple(sorted([ft, tt]))
            edge_map[key].append(r)

    for (t1, t2), rels in edge_map.items():
        # Use first relationship's direction
        first = rels[0]
        label = first.get("cardinality", "")
        if len(rels) > 1:
            label += f" ({len(rels)} rels)"

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
        # Fact tables
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
                "font-size": "9px",
                "color": "#94a3b8",
                "text-rotation": "autorotate",
                "text-margin-y": -10,
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
    ]


# =============================================================================
# SUMMARY PANEL BUILDER
# =============================================================================

def build_summary_panel(model):
    """Build full-width summary section below ERD with card layout."""
    if not model:
        return html.Div("No model generated yet.", style={"color": "#94a3b8", "padding": "20px"})

    sections = []

    # --- Business Data Summary card ---
    summary = model.get("data_summary", "")
    if summary:
        sections.append(
            html.Div([
                html.H3(["\U0001f4cb Business Data Summary"], className="card-title"),
                html.P(summary, style={"color": "#334155", "lineHeight": "1.8", "fontSize": "14px"}),
            ], className="summary-card")
        )

    # --- Recommended Consumers card ---
    consumers = model.get("recommended_consumers", [])
    if consumers:
        badge_colors = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed", "#0891b2", "#be185d", "#ca8a04"]
        badges = []
        for i, c in enumerate(consumers):
            bg = badge_colors[i % len(badge_colors)]
            badges.append(html.Span(c, style={
                "display": "inline-block", "padding": "6px 16px", "borderRadius": "20px",
                "background": bg, "color": "white", "fontSize": "13px", "fontWeight": "500",
                "margin": "4px",
            }))
        sections.append(
            html.Div([
                html.H3(["\U0001f465 Recommended Consumers"], className="card-title"),
                html.Div(badges, style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
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
            ttype = t.get("table_type", "dimension")
            cols = t.get("columns", [])
            pks = sum(1 for c in cols if c.get("role", "").lower() == "pk")
            fks = sum(1 for c in cols if c.get("role", "").lower() == "fk")
            measures = sum(1 for c in cols if c.get("role", "").lower() == "measure")
            sources = ", ".join(t.get("source_tables", []))
            rows.append(html.Tr([
                html.Td(html.Span(ttype.upper(), className=f"badge badge-{ttype}")),
                html.Td(t.get("table_name", "?"), style={"fontWeight": "600"}),
                html.Td(t.get("description", ""), style={"color": "#475569", "fontSize": "13px"}),
                html.Td(str(len(cols)), style={"textAlign": "center", "fontWeight": "600", "color": "#2563eb"}),
                html.Td(f"{pks}PK / {fks}FK / {measures}M", style={"fontSize": "12px", "color": "#64748b"}),
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
    consumers = m.get("recommended_consumers", [])
    catalog = m.get("_catalog", "")
    schema = m.get("_schema", "")
    total_cols = sum(len(t.get("columns", [])) for t in tables)
    type_order = {"fact": 0, "dimension": 1, "bridge": 2, "aggregate": 3}
    tables_sorted = sorted(tables, key=lambda t: type_order.get(t.get("table_type", "").lower(), 9))
    type_counts = {}
    for t in tables:
        tt = t.get("table_type", "dimension").lower()
        type_counts[tt] = type_counts.get(tt, 0) + 1
    CLR = {"fact": (0, 82, 155), "dimension": (0, 128, 80), "bridge": (180, 90, 0), "aggregate": (100, 50, 160)}

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
    pdf.cell(W, 6, "DIMENSIONAL DATA MODEL")
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
    section_title("01", "Executive Summary")
    body_text(summary_text or "No data summary available.", 9)
    if consumers:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(W, 5, "Recommended Data Consumers", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for c in consumers:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*DGRAY)
            pdf.cell(W, 4.5, f"    {c}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

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
    body_text("The following table inventory summarizes all proposed tables in the dimensional model, "
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
        ttype = t.get("table_type", "dimension").lower()
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
        rl = {"pk": "PK", "fk": "FK", "measure": "Measure", "attribute": "Attr"}
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
        body_text(f"{len(dropped)} columns were excluded from the dimensional model.", 8)
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
    label_value("Report Type:", "Dimensional Data Model Design - Consulting Deliverable")
    label_value("Source Schema:", f"{catalog}.{schema}" if catalog else "N/A")
    label_value("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    label_value("Proposed Tables:", str(len(tables)))
    label_value("Total Columns:", str(total_cols))
    label_value("Relationships:", str(len(rels)))
    label_value("Dropped Columns:", str(len(dropped)))
    label_value("Table Types:", ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items())))
    label_value("Consumers:", ", ".join(consumers) if consumers else "N/A")
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
                    html.Label("LAYOUT", className="control-label"),
                    dcc.Dropdown(
                        id="layout-dd",
                        options=[
                            {"label": "Cola", "value": "cola"},
                            {"label": "Dagre", "value": "dagre"},
                            {"label": "Breadthfirst", "value": "breadthfirst"},
                            {"label": "Circle", "value": "circle"},
                            {"label": "Grid", "value": "grid"},
                            {"label": "Concentric", "value": "concentric"},
                        ],
                        value="cola",
                        className="dash-dropdown", style={"width": "130px"},
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
                            layout={"name": "cola", "animate": False, "nodeSpacing": 80,
                                    "edgeLengthVal": 150, "maxSimulationTime": 2000},
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

        # Legend bar
        html.Div(
            [
                html.Span("\u25cf", style={"color": COLORS["fact"], "marginRight": "4px"}),
                html.Span("Fact", style={"marginRight": "16px"}),
                html.Span("\u25cf", style={"color": COLORS["dimension"], "marginRight": "4px"}),
                html.Span("Dimension", style={"marginRight": "16px"}),
                html.Span("\u25cf", style={"color": COLORS["bridge"], "marginRight": "4px"}),
                html.Span("Bridge", style={"marginRight": "16px"}),
                html.Span("\u25cf", style={"color": COLORS["aggregate"], "marginRight": "4px"}),
                html.Span("Aggregate", style={"marginRight": "16px"}),
                html.Div(style={"flex": "1"}),
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
    State("layout-dd", "value"),
    prevent_initial_call=True,
)
def on_generate(n_clicks, catalog, schema, layout):
    if not catalog or not schema:
        return {"display": "none"}, True

    token = _get_token()
    _reset_job()

    _job["_catalog"] = catalog
    _job["_schema"] = schema
    thread = threading.Thread(
        target=_bg_generate,
        args=(catalog, schema, layout, token),
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
            elements = build_proposed_erd_elements(model)
            summary = build_summary_panel(model)
            return step_items, elapsed, elements, summary, model, False, {"display": "none"}, True
        elif _job.get("error"):
            err_msg = _job["error"]
            error_panel = html.Div(
                [html.H4("Error"), html.P(err_msg, style={"color": "#dc2626"})],
                className="summary-card",
            )
            return step_items, elapsed, [], error_panel, None, True, {"display": "none"}, True

    # Still in progress
    return step_items, elapsed, no_update, no_update, no_update, no_update, no_update, no_update


# --- Layout switcher ---
@app.callback(
    Output("erd-graph", "layout"),
    Input("layout-dd", "value"),
)
def on_layout_change(layout_name):
    layout_config = {"name": layout_name or "cola"}
    if layout_name == "cola":
        layout_config.update({"animate": False, "nodeSpacing": 80, "edgeLengthVal": 150,
                              "maxSimulationTime": 2000})
    elif layout_name == "dagre":
        layout_config.update({"animate": False, "rankDir": "TB", "nodeSep": 60, "rankSep": 80})
    else:
        layout_config.update({"animate": False})
    return layout_config


# --- Search filter ---
@app.callback(
    Output("erd-graph", "stylesheet"),
    Input("search-input", "value"),
)
def on_search(query):
    base = get_cytoscape_stylesheet()
    if not query or not query.strip():
        return base

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

    ttype = table_info.get("table_type", "dimension")
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

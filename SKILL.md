# AI Data Model Designer — Complete Build Guide

> **Purpose**: Complete blueprint to rebuild the AI Data Model Designer app from scratch in any Databricks workspace.
>
> **Last Updated**: 2026-05-02
> **Owner**: sunny.singh@databricks.com

---

## 1. Project Overview

**What it is**: A Databricks App (Dash/Cytoscape) called **"AI Data Model Designer"** that:
1. Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
2. Profiles schema data via TABLESAMPLE (10 rows per table)
3. Uses a **Map-Reduce LLM pattern**: Sonnet for parallel entity extraction (Map), Opus for global relationship inference (Reduce)
4. Supports **dual model types**: 3NF Relational or Dimensional (Kimball Star Schema)
5. Renders a color-coded interactive ERD graph with professional styling
6. Exports a consulting-grade multi-page PDF report

**App Name**: `ai-data-model-designer`
**Title**: "AI Data Model Designer"
**Cloud**: Azure Databricks

---

## 2. File Structure

```
/Workspace/Users/<user>/ai-data-model-designer-git/
├── INSTALL.ipynb               # One-click installer notebook
├── README.md                   # Project documentation
├── SKILL.md                    # This file
├── app/                        # Databricks App (deployed from here)
│   ├── app.py                  # Main Dash application (~2,970 lines, ~123KB, 60 functions)
│   ├── app.yaml                # Databricks App config
│   ├── requirements.txt        # Python dependencies (dash, dash-cytoscape, databricks-sdk, pandas, fpdf2)
│   └── assets/
│       ├── style.css           # Custom CSS (~7.4KB: badges for 8 entity types, radio buttons, legend)
│       ├── logo.png            # Company logo (UI header + PDF)
│       └── logo_data.js        # Base64-encoded logo for PDF export
```

### app.yaml
```yaml
command:
  - python
  - app.py
env:
  - name: DATABRICKS_WAREHOUSE_ID
    valueFrom: sql-warehouse
```

### requirements.txt
```
dash>=2.14.0
dash-cytoscape>=1.0.0
databricks-sdk>=0.20.0
pandas>=1.5.0
```

---

## 3. App Resources & Authorization

### Resources (configured via SDK/UI)
| Resource Key | Type | Permission | Purpose |
|---|---|---|---|
| `sql-warehouse` | SQL Warehouse | CAN_USE | Execute profiling SQL |
| `serving-endpoint` | Serving Endpoint | CAN_QUERY | LLM API calls |

### User API Scopes
| Scope | Purpose |
|---|---|
| `sql` | Execute SQL on warehouse |
| `serving.serving-endpoints` | Foundation Model API |
| `catalog.catalogs:read` | List catalogs in dropdown |
| `catalog.schemas:read` | List schemas in dropdown |
| `catalog.tables:read` | Read information_schema |

---

## 4. Architecture — Map-Reduce LLM Pattern

```
User selects Catalog + Schema + Model Type (3NF / Dimensional) → clicks Generate
  │
  ├─ Captures OAuth token from Flask request headers
  ├─ Starts daemon thread: _bg_generate(catalog, schema, model_type, token)
  ├─ Shows real-time thinking modal (dcc.Interval polls every 1.2s)
  │
  ├─ Step 1: METADATA
  │   └─ SQL: information_schema.columns → table/column list
  │
  ├─ Step 2: PROFILING (ThreadPoolExecutor, 20 workers)
  │   └─ Per table: ONE query with TABLESAMPLE (10 ROWS)
  │       ├─ COUNT(*), COUNT(DISTINCT col), NULL counts
  │       ├─ MIN/MAX for date/timestamp columns
  │       └─ 3 sample values for top 4 string columns
  │
  ├─ Step 3: MAP PHASE (Sonnet, parallel)
  │   ├─ _build_per_table_payloads(): individual payloads from profiles (NO truncation)
  │   ├─ Groups of 3 tables (LLM_GROUP_SIZE=3)
  │   ├─ 20 workers, round-robin across latest Sonnet endpoint(s)
  │   ├─ On 400 BAD_REQUEST (oversized): auto-split into individual table calls
  │   └─ Programmatic merge: Python dedup by table_name (<1ms)
  │
  ├─ Step 4: REDUCE PHASE (Opus, racing)
  │   ├─ _build_schema_catalog(): compressed structure-only catalog (~400 chars/table)
  │   ├─ _build_reduce_prompt(): focused FK→PK inference prompt
  │   ├─ _call_chat_model_race(): 3 Opus endpoints racing
  │   └─ Validates: drops relationships to non-existent tables/columns
  │   └─ Injects FK roles into columns
  │
  ├─ Step 5: PARSE & VALIDATE
  │   └─ _parse_llm_json(raw, model_type): type/role normalization, fallback logic
  │
  └─ Step 6: RENDER
      ├─ build_proposed_erd_elements() → Cytoscape nodes + edges
      ├─ Cola layout with randomize=true, fit=true
      └─ Dynamic legend updates based on model_type
```

### Tiered Model Pools (_get_model_pools)

| Phase | Primary | Fallback | Rationale |
|---|---|---|---|
| **Map** | Latest Sonnet only (version-filtered) | Older Sonnet → Opus → Haiku | Fast, cheap structured extraction |
| **Reduce** | Top 3 Opus (racing) | Sonnet → Haiku | High-stakes cross-entity reasoning |

### Endpoint Discovery & Version Sorting

`_discover_endpoints()` queries all serving endpoints, filters for `claude` in name, and sorts by:
1. **Tier**: Opus (0) → Sonnet (1) → Haiku (2)
2. **Version** (descending): Versions are **padded to 3 digits** for correct comparison — e.g. `sonnet-4-6` → `(-4, -6, 0)` sorts before `sonnet-4` → `(-4, 0, 0)`, ensuring the latest sub-version is always selected.

**CRITICAL**: Without padding, Python tuple comparison treats `(-4,)` as less than `(-4, -6)`, making `sonnet-4` (old) sort before `sonnet-4-6` (latest). The 3-digit padding fixes this.

`_get_model_pools()` then filters the Sonnet list to only the **single latest version** (e.g., `sonnet-4-6`). Older Sonnet versions are pushed to the fallback chain.

Auto-refreshes every 5 minutes.

### Throughput (200 tables)
| Config | Map Time | Reduce Time | Total |
|---|---|---|---|
| Old (all Opus, group_size=1) | \~11 min | N/A | \~11 min |
| New (Sonnet Map + Opus Reduce) | \~1.5 min | \~45s | **\~2.5 min** |

---

## 5. Dual Model Types

### Model Type Selection
- **UI**: `dcc.RadioItems(id="model-type-radio")` with options: `3nf` (default), `dimensional`
- **Flow**: `on_generate` → `_bg_generate(model_type)` → prompts/parser/colors all parameterized

### 3NF Relational Mode
- Entity types: `core_entity`, `weak_entity`, `associative`, `reference`
- Column roles (Map): `pk`, `attribute` (FKs deferred to Reduce)
- Prompts: "Normalize to strict 3NF. Eliminate transitive dependencies."
- Reduce: FK→PK inference for relational integrity

### Dimensional (Kimball) Mode
- Entity types: `fact`, `dimension`, `bridge`, `aggregate`
- Column roles (Map): `pk`, `measure`, `attribute` (FKs deferred to Reduce)
- Prompts: "Design using Kimball methodology. Always include dim_date."
- Naming convention: `fact_`, `dim_`, `bridge_`, `agg_` prefixes
- Reduce: Fact→Dimension joins, bridge links

### Parameterized Functions
| Function | model_type behavior |
|---|---|
| `_build_system_prompt(mt)` | "3NF specialist" vs "Kimball methodology expert" |
| `_build_analysis_prompt(mt)` | Different rules, entity types, role options, JSON schema |
| `_build_group_prompt(mt)` | Same as analysis, for groups of 3 tables |
| `_build_reduce_prompt(mt)` | "relational integrity" vs "star schema joins" |
| `_parse_llm_json(raw, mt)` | Valid types/roles differ; fallback logic differs |
| `build_summary_panel(model)` | Shows Measures count for dimensional |
| PDF `_generate_consulting_report(m)` | Title, colors, type_order, role_labels all conditional |

---

## 6. Authentication Pattern

### Token Flow (CRITICAL for background threads)
```python
# Main request context (Flask callback):
token = flask_request.headers.get("x-forwarded-access-token")

# Passed to background thread:
threading.Thread(target=_bg_generate, args=(catalog, schema, model_type, token))

# Background thread sets thread-local:
_tls.token = token

# Profile workers receive token EXPLICITLY:
def _profile_single_table(catalog, schema, tname, cols, token=None):
    _tls.token = token
    rows = run_sql(sql, token=token)

# run_sql() handles None data_array:
data = resp.result.data_array if resp.result and resp.result.data_array else []
```

**KEY**: `flask_request.headers.get(...)` RAISES `RuntimeError` in background threads. Must wrap in `try/except RuntimeError`.

---

## 7. LLM Configuration

### Endpoint Discovery (_discover_endpoints)
- Queries all serving endpoints, filters for `claude` in name
- Sorts: Opus (tier 0) > Sonnet (tier 1) > Haiku (tier 2), version descending
- **Version padding**: Tuples padded to 3 digits so `sonnet-4-6` → `(-4,-6,0)` sorts before `sonnet-4` → `(-4,0,0)`
- Auto-refreshes every 5 minutes

### _call_chat_model(prompt, token, max_tokens, model_chain)
- REST API: `POST /serving-endpoints/{model}/invocations`
- On 429: exponential backoff (15s → 30s → 60s), Opus gets 3 attempts
- On 500/502/503: fall to next model
- On 400 + prompt > 100K chars: raise `OverflowError` (triggers group split)
- On 400 (other): raise `RuntimeError`

### _call_chat_model_race(prompt, token, max_tokens)
- Fires same prompt to all 3 Opus endpoints simultaneously
- Returns first successful response, abandons losers
- Used for: Reduce phase

### Oversized Group Recovery
When a group of 3 tables returns 400 (payload too large):
1. `_analyze_one_group` catches `OverflowError`
2. Splits group into individual table calls
3. Each table gets its own LLM call
4. Results merged back into single JSON response
5. No tables are lost

### Parameters
| Parameter | Value |
|---|---|
| LLM_WORKERS | 20 |
| LLM_GROUP_SIZE | 3 |
| PROFILE_WORKERS | 20 |
| MAX_TOKENS_SINGLE | 32,000 |
| MAX_TOKENS_GROUP | 16,000 |
| API_TIMEOUT | 300s |

---

## 8. Profiling

### Per-Table SQL (single query with TABLESAMPLE)
```sql
SELECT
  COUNT(*) AS __row_count__,
  COUNT(DISTINCT `col1`) AS dist__col1,
  SUM(CASE WHEN `col1` IS NULL THEN 1 ELSE 0 END) AS nulls__col1,
  CAST(MIN(`date_col`) AS STRING) AS min__date_col,
  CAST(MAX(`date_col`) AS STRING) AS max__date_col,
  (SELECT CONCAT_WS('|||', COLLECT_LIST(val))
   FROM (SELECT DISTINCT CAST(`str_col` AS STRING) AS val
         FROM table WHERE `str_col` IS NOT NULL LIMIT 3)) AS sample__str_col
FROM `catalog`.`schema`.`table` TABLESAMPLE (10 ROWS)
```

### Per-Table Payloads (NO truncation)
- `_profile_schema()` returns raw profiles list (not concatenated string)
- `_build_per_table_payloads()` builds individual payloads per table
- Each payload: header + table name + column profiles
- ALL profiled tables reach the LLM (old 500K truncation eliminated)

---

## 9. UI Structure

### Layout (top to bottom)
1. **Controls bar**: Catalog dropdown, Schema dropdown, **Model Type radio** (3NF / Dimensional), Generate button
2. **Main area**: Cytoscape graph (full width, `calc(100vh - 160px)`)
3. **Legend bar**: Dynamic entity type dots (updates on model_type) + Export Report button
4. **Summary panel**: Full width below graph
5. **Zoom controls**: +/-/fit buttons overlaid on graph

### Model Type Radio
```python
dcc.RadioItems(
    id="model-type-radio",
    options=[
        {"label": " 3NF Relational", "value": "3nf"},
        {"label": " Dimensional (Star)", "value": "dimensional"},
    ],
    value="3nf", inline=True,
)
```

### Dynamic Legend
- `update_legend()` callback: fires on `model-store` change
- Returns colored dots + labels for the active model type
- Export button is a SIBLING of legend-bar (not inside it) to avoid Dash callback conflicts

### Color Scheme
| 3NF Type | Color | Dimensional Type | Color |
|---|---|---|---|
| Core Entity | #1e3a8a (Navy) | Fact | #2563eb (Blue) |
| Weak Entity | #0ea5e9 (Sky Blue) | Dimension | #16a34a (Green) |
| Associative | #f97316 (Orange) | Bridge | #ea580c (Deep Orange) |
| Reference | #64748b (Slate) | Aggregate | #7c3aed (Purple) |

### Graph Configuration
| Setting | Value |
|---|---|
| Layout | Cola (force-directed) |
| randomize | **true** (prevents linear chain convergence) |
| nodeSpacing | 100 |
| edgeLengthVal | 200 |
| maxSimulationTime | 4000ms |
| convergenceThreshold | 0.001 |
| fit | true |
| padding | 50 |
| animate | **false** |
| Node shape | round-rectangle, 220×60px |
| Node font | Inter / Segoe UI, 12px, weight 600 |
| Node shadows | blur=12, offset-y=4 |
| Edge labels | text-background with roundrectangle shape |
| Edge consolidation | ONE edge per table pair |

### Visual Entity Differentiation
| Type | Border Style |
|---|---|
| Core Entity / Fact | Solid, thick (3-3.5px) |
| Weak Entity / Bridge | Dashed |
| Reference / Aggregate | Dotted |
| Associative / Dimension | Solid, medium |

---

## 10. Parser (_parse_llm_json)

### Model-Type Aware Parsing
```python
def _parse_llm_json(raw, model_type="3nf"):
    # Valid types per model
    if model_type == "dimensional":
        valid_types = {"fact", "dimension", "bridge", "aggregate"}
    else:
        valid_types = {"core_entity", "weak_entity", "associative", "reference"}
    
    # Valid roles per model
    valid_roles = {"pk", "fk", "measure", "attribute"} if dimensional else {"pk", "fk", "attribute"}
```

### Fallback Logic (invalid types)
- **3NF**: "bridge/link/junction" in name → associative, "ref/lkp/lookup" → reference, "_dep/detail/line" → weak_entity, else core_entity
- **Dimensional**: "fact/fct" in name → fact, "bridge/link/junction" → bridge, "agg/summary" → aggregate, else dimension

---

## 11. PDF Export

### Server-Side Generation (fpdf2)
- `_generate_consulting_report(model)` produces McKinsey-style report
- Model-type aware: title, colors, type_order, role_labels all conditional
- `_deep_sanitize()` cleans Unicode → ASCII for PDF compatibility

### Conditional Elements by Model Type
| Element | 3NF | Dimensional |
|---|---|---|
| Title | "RELATIONAL DATA MODEL (3NF)" | "DIMENSIONAL DATA MODEL" |
| Type Order | core_entity → weak_entity → associative → reference | fact → dimension → bridge → aggregate |
| Role Labels | PK, FK, Attr | PK, FK, Measure, Attr |
| Body Text | "3NF relational model" | "dimensional model (Kimball star schema)" |

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| Sonnet for Map, Opus for Reduce | Entity extraction is structurally simple (Sonnet is 3-5x faster). FK inference needs highest intelligence. |
| group_size=3 | Balances parallelism (67 groups / 200 tables) with context richness |
| Per-table payloads (no concat) | Eliminates the 500K MAX_PAYLOAD_CHARS truncation that silently dropped tables |
| Programmatic merge (no LLM merge) | Instant, deterministic. Each Map call already has all_table_names context. |
| Compressed catalog for Reduce | Only structure needed for FK inference. 200 tables ≈ 80K chars ≈ 20K tokens. |
| Race mode for Reduce | Single high-stakes call. Racing 3 endpoints ensures fastest response. |
| randomize: true on Cola | Without it, all nodes start at (0,0) and converge to a vertical line. |
| OverflowError → group split | On 400 BAD_REQUEST, split oversized group into individual calls instead of failing. |
| Export button outside legend-bar | Prevents Dash destroying/recreating the button when legend callback fires. |
| Padded version sorting (3-digit) | Ensures `sonnet-4-6` correctly beats `sonnet-4` as "latest". Without padding, Python treats `(-4,)` < `(-4,-6)`. |

---

## 13. Common Bugs & Fixes

| Bug | Cause | Fix |
|---|---|---|
| "Working outside of request context" | flask_request in background thread | try/except RuntimeError |
| Only 35/116 tables sent to LLM | _build_profiling_payload truncated at 500K | _build_per_table_payloads (no truncation) |
| Graph renders as vertical chain | Cola nodes all start at (0,0) | randomize: true |
| 400 BAD_REQUEST on large groups | Group prompt exceeds context limit | OverflowError → auto-split |
| "consumers" not defined in PDF | Leftover reference after removal | Removed all consumers references |
| App UNAVAILABLE after deploy | update_legend recreated export-btn | export-btn moved outside legend-bar |
| run_sql crashes on empty result | data_array is None | `data = resp.result.data_array if resp.result and resp.result.data_array else []` |
| Old Sonnet models used in Map | Version sorting puts `sonnet-4` before `sonnet-4-6` | Pad version tuples to 3 digits: `(-4,0,0)` vs `(-4,-6,0)` |
| Error label shows wrong model | Used LLM_FALLBACK_CHAIN for label | Use work_models instead |

---

## 14. Testing Checklist

- [ ] Select catalog/schema → tables populate
- [ ] Select **3NF Relational** → generates core_entity/weak_entity/associative/reference types
- [ ] Select **Dimensional (Star)** → generates fact/dimension/bridge/aggregate types
- [ ] All profiled tables appear in Map phase (no truncation)
- [ ] Reduce phase shows relationships mapped via Opus
- [ ] Cola layout renders proper 2D spread (not vertical chain)
- [ ] Legend updates when switching model type
- [ ] Click nodes → detail panel shows columns with role badges
- [ ] Summary panel shows Measures count for dimensional
- [ ] Export Report → PDF downloads with correct title/colors
- [ ] Kill switch (✕ Stop) aborts in-progress generation
- [ ] Test on small schema (3 tables) and large schema (100+ tables)
- [ ] Verify Map phase uses latest Sonnet (e.g. sonnet-4-6, not sonnet-4)

---

## 15. Known Limitations

1. Views don't support DESCRIBE DETAIL — row count estimated from sample
2. 200K token context handles ~200 tables max in Reduce phase
3. No DDL generation (proposes model but doesn't create tables)
4. Single-schema scope (no cross-schema relationships)
5. Tables with 0 sample rows are excluded from Map phase (no data to model)

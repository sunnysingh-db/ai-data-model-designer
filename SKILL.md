# AI Data Model Designer — Complete Build Guide

> **Purpose**: Complete blueprint to rebuild the AI Data Model Designer app from scratch in any Databricks workspace.
>
> **Last Updated**: 2026-05-04
> **Owner**: sunny.singh@databricks.com

---

## 1. Project Overview

**What it is**: A Databricks App (Dash/Cytoscape) called **"AI Data Model Designer"** that:
1. Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
2. Profiles schema data via TABLESAMPLE (10 rows per table)
3. Uses a **Map-Reduce LLM pattern**: Sonnet for parallel entity extraction (Map), best Opus for global relationship inference (Reduce)
4. Applies **multi-layer relationship detection**: LLM Reduce + heuristic FK matching (exact + suffix) + LLM rescue pass for orphaned entities
5. Supports **dual model types**: 3NF Relational or Dimensional (Kimball Star Schema)
6. Renders a **domain-clustered interactive ERD graph** with connected component detection and compound parent nodes
7. Exports a consulting-grade multi-page PDF report

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
│   ├── app.py                  # Main Dash application (~3,200 lines, ~135KB)
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
  │   ├─ Naming convention rules enforced (singular, snake_case, <entity>_id PKs)
  │   ├─ On 400 BAD_REQUEST (oversized): auto-split into individual table calls
  │   └─ Programmatic merge: Python dedup by table_name (<1ms)
  │
  ├─ Step 4: REDUCE PHASE (Best Opus, single call)
  │   ├─ _build_schema_catalog(): compressed structure-only catalog (~400 chars/table)
  │   ├─ _build_reduce_prompt(): focused FK→PK inference prompt
  │   ├─ _call_chat_model(): single best Opus (no racing — saves 2/3 cost)
  │   ├─ Fallback chain: next Opus → Sonnet → Haiku
  │   └─ Validates: drops relationships to non-existent tables/columns
  │
  ├─ Step 5: HEURISTIC FK DETECTION (deterministic, instant)
  │   ├─ _infer_heuristic_fks(proposed_tables, existing_rels)
  │   ├─ Builds PK index: defaultdict(list) mapping column → ALL parent tables
  │   ├─ EXACT match: X.restaurant_id = restaurant.restaurant_id
  │   ├─ SUFFIX match: X.origin_airport_id ends with airport.airport_id
  │   └─ Deduplicates against existing relationships
  │
  ├─ Step 6: LLM RESCUE PASS (Sonnet, targeted)
  │   ├─ _rescue_orphans_via_llm(proposed_tables, existing_rels, token)
  │   ├─ Identifies orphans: entities with 0 relationships after Reduce + heuristic
  │   ├─ Builds mini-catalog: full details for orphans, PK-only for connected
  │   ├─ Sonnet call (~5K chars prompt, fast/cheap)
  │   └─ Catches semantic matches: origin_airport→airport, airline_code→airline
  │
  ├─ Step 7: PARSE & VALIDATE
  │   └─ _parse_llm_json(raw, model_type): type/role normalization, fallback logic
  │
  └─ Step 8: RENDER
      ├─ _detect_domain_clusters(): BFS connected component detection
      ├─ build_proposed_erd_elements(): compound parent nodes per cluster
      ├─ fcose layout with dynamic scaling (nodeRepulsion, componentSpacing)
      └─ Dynamic legend updates based on model_type
```

### Tiered Model Pools (_get_model_pools)

| Phase | Primary | Fallback | Rationale |
|---|---|---|---|
| **Map** | Latest Sonnet only (version-filtered) | Older Sonnet → Opus → Haiku | Fast, cheap structured extraction |
| **Reduce** | Best single Opus (no racing) | Next Opus → Sonnet → Haiku | High-intelligence reasoning; cost-efficient |
| **Rescue** | Latest Sonnet (via map pool) | Older Sonnet → Opus → Haiku | Cheap/fast for small targeted prompt |

### Multi-Layer Relationship Detection

| Layer | Function | Type | What It Catches | Example |
|---|---|---|---|---|
| 1. Opus Reduce | `_map_global_relationships` | LLM | Cross-entity FK→PK via reasoning | `order.customer_id → customer.customer_id` |
| 2. Heuristic (Exact) | `_infer_heuristic_fks` | Deterministic | Column name = PK name | `payment.order_id → order.order_id` |
| 3. Heuristic (Suffix) | `_infer_heuristic_fks` | Deterministic | Column ends with `_` + PK name | `flight.origin_airport_id → airport.airport_id` |
| 4. LLM Rescue | `_rescue_orphans_via_llm` | LLM (Sonnet) | Semantic matches for non-standard naming | `flights.airline_code → airline.iata_code` |

### PK Index Design (CRITICAL)

The heuristic FK detection uses `defaultdict(list)` for the PK index:

```python
pk_index = defaultdict(list)  # column_name -> [table1, table2, ...]
for t in proposed_tables:
    for c in t.get("columns", []):
        if c.get("role", "").lower() == "pk":
            pk_index[c["name"]].append(t["table_name"])
```

**Why not a plain dict?** With 116 tables, multiple tables may share the same PK column name (e.g., both `orders.order_id` and `vcom_orders.order_id` are PKs). A plain dict would overwrite `"orders"` with `"vcom_orders"` (last-one-wins), causing `payments.order_id` to link to the wrong table.

### Endpoint Discovery & Version Sorting

`_discover_endpoints()` queries all serving endpoints, filters for `claude` in name, and sorts by:
1. **Tier**: Opus (0) → Sonnet (1) → Haiku (2)
2. **Version** (descending): Versions are **padded to 3 digits** for correct comparison — e.g. `sonnet-4-6` → `(-4, -6, 0)` sorts before `sonnet-4` → `(-4, 0, 0)`, ensuring the latest sub-version is always selected.

**CRITICAL**: Without padding, Python tuple comparison treats `(-4,)` as less than `(-4, -6)`, making `sonnet-4` (old) sort before `sonnet-4-6` (latest). The 3-digit padding fixes this.

`_get_model_pools()` then filters the Sonnet list to only the **single latest version** (e.g., `sonnet-4-6`). Older Sonnet versions are pushed to the fallback chain.

Auto-refreshes every 5 minutes.

### Progress Callbacks

The `_map_global_relationships` function accepts a `progress_cb` parameter (wired to `_add_step` in `_bg_generate`). This emits 8 UI progress steps throughout the Reduce phase:

1. 🧠 Calling opus-4-7 for global FK→PK mapping...
2. 🏆 opus-4-7 responded — X chars
3. 🔍 Parsing relationship JSON...
4. ✅ Validating X proposed relationships against entity schema...
5. ✅ X valid LLM relationships (Y dropped)
6. 🔗 Running heuristic FK detection (exact + suffix matching)...
7. 🔍 Scanning for orphaned entities...
8. 🚑 Rescue pass connected X orphaned entities

**Why this matters**: Without these callbacks, the UI shows no updates between "Opus responded" and "Reduce complete" — a gap of 10-60 seconds that makes the app appear stuck.

### Throughput (200 tables)
| Config | Map Time | Reduce Time | Total |
|---|---|---|---|
| Old (all Opus, group_size=1) | ~11 min | N/A | ~11 min |
| New (Sonnet Map + single Opus Reduce) | ~1.5 min | ~45s | **~2.5 min** |

---

## 5. Dual Model Types

### Model Type Selection
- **UI**: `dcc.RadioItems(id="model-type-radio")` with options: `3nf` (default), `dimensional`
- **Flow**: `on_generate` → `_bg_generate(model_type)` → prompts/parser/colors all parameterized

### 3NF Relational Mode
- Entity types: `core_entity`, `weak_entity`, `associative`, `reference`
- Column roles (Map): `pk`, `attribute` (FKs deferred to Reduce)
- Prompts: "Normalize to strict 3NF. Eliminate transitive dependencies."
- **Naming conventions**: Singular nouns, snake_case, `<entity>_id` PKs, `_type`/`_status` suffixes for reference tables
- Reduce: FK→PK inference for relational integrity

### Dimensional (Kimball) Mode
- Entity types: `fact`, `dimension`, `bridge`, `aggregate`
- Column roles (Map): `pk`, `measure`, `attribute` (FKs deferred to Reduce)
- Prompts: "Design using Kimball methodology. Always include dim_date."
- Naming convention: `fact_`, `dim_`, `bridge_`, `agg_` prefixes
- Reduce: Fact→Dimension joins, bridge links

### Naming Conventions (enforced in prompts)

Both `_build_analysis_prompt` and `_build_group_prompt` include Rule 3 — NAMING CONVENTIONS:
- Singular nouns (`order`, not `orders`; `restaurant`, not `restaurants`)
- snake_case, no abbreviations
- PK: `<entity>_id` (e.g. `order_id` for `order` entity)
- Reference tables: `_type`/`_status` suffix (e.g. `order_status`)
- FK columns match parent PK name

**Dual benefit**: Improves table names for users AND makes the suffix heuristic more effective (ensures columns follow `_id` pattern).

### Parameterized Functions
| Function | model_type behavior |
|---|---|
| `_build_system_prompt(mt)` | "3NF specialist" vs "Kimball methodology expert" + RECOMMEND guidance |
| `_build_analysis_prompt(mt)` | Different rules, entity types, role options, naming conventions, JSON schema |
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
- Fires same prompt to selected endpoints simultaneously
- Returns first successful response, abandons losers via `pool.shutdown(wait=False, cancel_futures=True)`
- Background timer updates UI every 10s during race
- **No longer used for Reduce phase** — Reduce now uses single `_call_chat_model` call
- Still available for other use cases if needed

### Cost Optimization (Single Opus for Reduce)
- **Old**: `_call_chat_model_race` fired 3 Opus endpoints simultaneously, paying for all 3
- **New**: `_call_chat_model` with `model_chain=[best_opus] + fallback`
- **Savings**: 2/3 reduction in Opus token cost per Reduce call
- Fallback chain still handles errors (if opus-4-7 fails, tries next Opus, then Sonnet)

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

## 9. Heuristic FK Detection

### _infer_heuristic_fks(proposed_tables, existing_rels)

Deterministic, instant (<1ms). Runs after Opus Reduce to catch obvious matches the LLM missed.

**Strategy 1 — EXACT match**:
```
For each table T, for each column C where C.name == PK.name in another table:
  → T.C → Parent.PK (if not self-referencing and not already in existing_rels)
```
Example: `payment.order_id` = `order.order_id` → `payment.order_id → order.order_id`

**Strategy 2 — SUFFIX match**:
```
For each table T, for each column C where C.name ends with "_" + PK.name:
  → T.C → Parent.PK (if len(C.name) > len(PK.name) + 1)
```
Example: `flight.origin_airport_id` ends with `_airport_id` → `flight.origin_airport_id → airport.airport_id`

**PK Index** (defaultdict(list)):
```python
pk_index = defaultdict(list)
for t in proposed_tables:
    for c in t.get("columns", []):
        if c.get("role", "").lower() == "pk":
            pk_index[c["name"]].append(t["table_name"])
```

Using `defaultdict(list)` is critical — a plain dict would overwrite when multiple tables share the same PK column name (e.g., `orders.order_id` and `vcom_orders.order_id`).

---

## 10. LLM Rescue Pass

### _rescue_orphans_via_llm(proposed_tables, existing_rels, token, model_type)

After Opus Reduce + heuristic, some entities may still have ZERO relationships — especially with non-standard naming (e.g., `origin_airport` instead of `airport_id`, `airline_code` instead of `airline_id`, SAP-style `Vendor_Code`).

**Flow**:
1. Identify orphans: entities in `all_table_names - connected`
2. Build mini-catalog: full column details for orphans, PK-only summary for connected entities
3. Call Sonnet with focused prompt (~5K chars) asking for semantic FK→PK relationships
4. Validate against entity schema (same logic as Opus validation)
5. Deduplicate against existing relationships

**Cost**: Much cheaper than full Reduce — typically <5K chars prompt vs 80K+ for the full catalog.

---

## 11. UI Structure

### Layout (top to bottom)
1. **Controls bar**: Catalog dropdown, Schema dropdown, **Model Type radio** (3NF / Dimensional), Generate button
2. **Main area**: Cytoscape graph (full width, `calc(100vh - 160px)`)
3. **Legend bar**: Dynamic entity type dots (updates on model_type) + Export Report button
4. **Summary panel**: Full width below graph
5. **Zoom controls**: +/-/fit buttons overlaid on graph

### Domain-Clustered Graph Layout

**Connected Component Detection** (`_detect_domain_clusters`):
- BFS traversal on relationship edges
- Each connected component = one domain cluster
- Isolated nodes (no relationships) get their own cluster

**Compound Parent Nodes**:
- Each cluster with >1 node gets a compound parent node
- Domain naming via keyword matching:
  - `flight`/`airline`/`airport` → "Aviation"
  - `order`/`restaurant`/`payment` → "Orders"
  - `procurement`/`contract`/`vendor` → "Procurement"
  - etc.
- Parent nodes styled: dashed border, light background, domain label at top

**fcose Layout** (replaced Cola):
- Natively handles compound nodes — places each cluster in its own region
- `componentSpacing` separates disconnected clusters
- **Dynamic scaling** based on entity count:
  ```python
  _n = len(elements)
  _repulsion = max(15000, _n * 200)   # More entities → more repulsion
  _spacing = max(400, _n * 5)          # More entities → more inter-cluster spacing
  _edge_len = max(250, _n * 3)         # More entities → longer edges
  ```

### Graph Configuration
| Setting | Value |
|---|---|
| Layout | **fcose** (force-directed with compound node support) |
| randomize | true |
| quality | "proof" (highest quality) |
| nodeRepulsion | 15000+ (dynamic: N×200) |
| idealEdgeLength | 250+ (dynamic: N×3) |
| componentSpacing | 400+ (dynamic: N×5) |
| nestingFactor | 0.1 |
| gravity | 0.08 |
| gravityRange | 5.0 |
| fit | true |
| padding | 80 |
| animate | false |
| Node shape | round-rectangle, 200×56px |
| Edge consolidation | ONE edge per table pair |

### Color Scheme
| 3NF Type | Color | Dimensional Type | Color |
|---|---|---|---|
| Core Entity | #1e3a8a (Navy) | Fact | #2563eb (Blue) |
| Weak Entity | #0ea5e9 (Sky Blue) | Dimension | #16a34a (Green) |
| Associative | #f97316 (Orange) | Bridge | #ea580c (Deep Orange) |
| Reference | #64748b (Slate) | Aggregate | #7c3aed (Purple) |

---

## 12. Parser (_parse_llm_json)

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

## 13. PDF Export

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

## 14. Key Design Decisions

| Decision | Rationale |
|---|---|
| Sonnet for Map, single Opus for Reduce | Entity extraction is structurally simple (Sonnet is 3-5x faster). FK inference needs highest intelligence. No racing saves 2/3 cost. |
| Multi-layer relationship detection | LLM + heuristic + rescue catches standard naming, suffix patterns, AND semantic matches |
| `defaultdict(list)` PK index | Handles schemas where multiple tables share the same PK column name (e.g., `order_id`) |
| Naming conventions in prompts | LLM recommends `<entity>_id` pattern, making suffix heuristic more effective |
| Domain-clustered fcose layout | Connected component detection groups related entities; compound nodes prevent cross-domain overlap |
| Dynamic layout scaling | `nodeRepulsion` and `componentSpacing` scale with entity count for large schemas |
| group_size=3 | Balances parallelism (67 groups / 200 tables) with context richness |
| Per-table payloads (no concat) | Eliminates the 500K MAX_PAYLOAD_CHARS truncation that silently dropped tables |
| Programmatic merge (no LLM merge) | Instant, deterministic. Each Map call already has all_table_names context. |
| Compressed catalog for Reduce | Only structure needed for FK inference. 200 tables ≈ 80K chars ≈ 20K tokens. |
| Progress callbacks in Reduce | 8 `_add_step` calls prevent UI stall between Opus response and final result |
| LLM rescue for orphans only | Cheap/fast — only processes disconnected entities, not the full 100+ table catalog |
| randomize: true on fcose | Without it, all nodes start at (0,0) and converge to a vertical line. |
| OverflowError → group split | On 400 BAD_REQUEST, split oversized group into individual calls instead of failing. |
| Export button outside legend-bar | Prevents Dash destroying/recreating the button when legend callback fires. |
| Padded version sorting (3-digit) | Ensures `sonnet-4-6` correctly beats `sonnet-4` as "latest". Without padding, Python treats `(-4,)` < `(-4,-6)`. |

---

## 15. Common Bugs & Fixes

| Bug | Cause | Fix |
|---|---|---|
| payments.order_id linked to wrong table | PK index used plain dict; `vcom_orders` overwrote `orders` | Changed to `defaultdict(list)` for multi-valued PK index |
| Airport entity disconnected from flights | `origin_airport_id` ≠ `airport_id` (no exact match) | Added SUFFIX matching: `origin_airport_id` ends with `_airport_id` |
| Non-standard FK naming missed (airline_code) | Heuristic only does pattern matching; semantic patterns need LLM | Added LLM rescue pass for orphaned entities |
| Graph nodes overlap across domains | Cola layout treats all nodes equally; no domain clustering | Switched to fcose with connected component detection + compound parent nodes |
| Progress stalls after Opus responds | No `_add_step` calls between Opus response and Reduce complete | Added 8 progress callbacks throughout `_map_global_relationships` |
| 3x Opus cost on Reduce | `_call_chat_model_race` fired 3 endpoints simultaneously | Changed to single `_call_chat_model` with fallback chain |
| LLM returns plural table names | No naming convention guidance in prompts | Added NAMING CONVENTIONS rule to system and analysis prompts |
| "Working outside of request context" | flask_request in background thread | try/except RuntimeError |
| Only 35/116 tables sent to LLM | _build_profiling_payload truncated at 500K | _build_per_table_payloads (no truncation) |
| Graph renders as vertical chain | Nodes all start at (0,0) | randomize: true |
| 400 BAD_REQUEST on large groups | Group prompt exceeds context limit | OverflowError → auto-split |
| App UNAVAILABLE after deploy | update_legend recreated export-btn | export-btn moved outside legend-bar |
| run_sql crashes on empty result | data_array is None | `data = resp.result.data_array if resp.result and resp.result.data_array else []` |
| Old Sonnet models used in Map | Version sorting puts `sonnet-4` before `sonnet-4-6` | Pad version tuples to 3 digits |
| Duplicate function definitions | Earlier edits left orphaned copies | Cleaned up — single copy of each function |

---

## 16. Testing Checklist

- [ ] Select catalog/schema → tables populate
- [ ] Select **3NF Relational** → generates core_entity/weak_entity/associative/reference types
- [ ] Select **Dimensional (Star)** → generates fact/dimension/bridge/aggregate types
- [ ] All profiled tables appear in Map phase (no truncation)
- [ ] Table names follow naming conventions (singular, snake_case, `<entity>_id` PKs)
- [ ] Reduce phase shows single Opus call (not racing)
- [ ] Progress updates throughout Reduce: validation, heuristic, rescue steps visible
- [ ] Heuristic FK detection finds exact + suffix matches
- [ ] LLM rescue pass connects orphaned entities
- [ ] Domain clusters visible: related entities grouped in dashed containers
- [ ] Unrelated domains (e.g., Aviation vs Orders) visually separated
- [ ] Layout scales properly for large schemas (100+ entities)
- [ ] Legend updates when switching model type
- [ ] Click nodes → detail panel shows columns with role badges
- [ ] Summary panel shows Measures count for dimensional
- [ ] Export Report → PDF downloads with correct title/colors
- [ ] Kill switch (✕ Stop) aborts in-progress generation
- [ ] Test on small schema (3 tables) and large schema (100+ tables)
- [ ] Verify Map phase uses latest Sonnet (e.g. sonnet-4-6, not sonnet-4)

---

## 17. Known Limitations

1. Views don't support DESCRIBE DETAIL — row count estimated from sample
2. 200K token context handles ~200 tables max in Reduce phase
3. No DDL generation (proposes model but doesn't create tables)
4. Single-schema scope (no cross-schema relationships)
5. Tables with 0 sample rows are excluded from Map phase (no data to model)
6. Domain cluster naming uses keyword heuristic — may not label every domain correctly
7. Rescue pass only targets entities with 0 relationships (partially-connected entities not rescued)

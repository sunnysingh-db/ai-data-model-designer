# Intelligent ERD App — Complete Build Guide

> **Purpose**: Complete blueprint to rebuild the AI Data Model Designer app from scratch in any Databricks workspace.
>
> **Last Updated**: 2026-04-27
> **Owner**: maintainer@your-org.com

---

## 1. Project Overview

**What it is**: A Databricks App (Dash/Cytoscape) called **"AI Data Model Designer"** that:
1. Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
2. Profiles schema data via TABLESAMPLE (10 rows per table)
3. Sends profiling data to Claude Opus LLM (3 endpoints racing simultaneously)
4. Receives an optimized star/snowflake dimensional model
5. Renders it as a color-coded interactive ERD graph
6. Exports a consulting-grade multi-page PDF report

**App Name**: `ai-data-model-designer`
**Title**: "AI Data Model Designer"
**Cloud**: Azure Databricks

---

## 2. File Structure

```
/Workspace/Users/<user>/ai-data-model-designer/
├── INSTALL.ipynb               # One-click installer notebook
├── README.md                   # Project documentation
├── SKILL.md                    # This file
├── app/                        # Databricks App (deployed from here)
│   ├── app.py                  # Main Dash application (~2,400 lines, ~97KB)
│   ├── app.yaml                # Databricks App config
│   ├── requirements.txt        # Python dependencies (dash, dash-cytoscape, databricks-sdk, pandas, pyyaml, fpdf2)
│   ├── assets/
│   │   ├── style.css           # Custom CSS (pulse animation for thinking modal)
│   │   ├── logo.png            # Company logo (used in UI header + PDF)
│   │   └── logo_data.js        # Base64-encoded logo for PDF export
│   └── scripts/
│       └── generate_logo_b64.py
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
| `serving.serving-endpoints` | Foundation Model API (**NOTE**: NOT `serving-endpoints`) |
| `catalog.catalogs:read` | List catalogs in dropdown |
| `catalog.schemas:read` | List schemas in dropdown |
| `catalog.tables:read` | Read information_schema |

---

## 4. Architecture — Complete Data Flow

```
User selects Catalog + Schema → clicks "🤖 Generate Data Model"
  │
  ├─ Captures OAuth token from Flask request headers
  ├─ Starts daemon thread: _bg_generate(catalog, schema, layout, token)
  ├─ Shows real-time thinking modal (dcc.Interval polls every 1.2s)
  │
  ├─ Step 1: METADATA (parent thread)
  │   └─ SQL: information_schema.columns → table/column list
  │
  ├─ Step 2: PROFILING (ThreadPoolExecutor, 20 workers)
  │   └─ Per table: ONE query with TABLESAMPLE (10 ROWS)
  │       ├─ COUNT(*), COUNT(DISTINCT col), NULL counts
  │       ├─ MIN/MAX for date/timestamp columns only
  │       ├─ 3 sample values for top 4 string columns
  │       └─ Row count: DESCRIBE DETAIL (Delta) or estimate (views)
  │   └─ Token passed EXPLICITLY to each worker (not threading.local)
  │
  ├─ Step 3: LLM ANALYSIS (ALWAYS DISTRIBUTE)
  │   ├─ group_size = 1 always (1 table per API call)
  │   ├─ Opus-only: opus-4-7 (primary) + opus-4-6 (fallback)
  │   ├─ Round-robin distribution across 2 Opus endpoints
  │   ├─ Fallback chain: opus-4-7 → opus-4-6 per call
  │   └─ PROGRAMMATIC MERGE: Python concatenate + deduplicate (<1s, no LLM merge)
  │
  ├─ Step 4: VALIDATION
  │   └─ parse_llm_model(): FK/PK integrity, orphan detection, type normalization
  │
  ├─ Step 5: RENDER
  │   └─ build_proposed_erd_elements() → Cytoscape nodes + edges
  │       └─ Consolidated edges: ONE arrow per table pair
  │
  └─ Poll callback picks up result → renders graph + summary panel
```

---

## 5. Authentication Pattern

### Token Flow (CRITICAL for background threads)
```python
# Main request context (Flask callback):
token = flask_request.headers.get("x-forwarded-access-token")

# Passed to background thread:
threading.Thread(target=_bg_generate, args=(catalog, schema, layout, token))

# Background thread sets thread-local:
_tls = threading.local()
_tls.token = token

# Profile workers receive token EXPLICITLY (threading.local doesn't inherit):
def _profile_single_table(catalog, schema, tname, cols, token=None):
    _tls.token = token  # Set for this worker thread
    ...
    rows = run_sql(sql, token=token)  # Pass explicitly too

# run_sql() priority: explicit param > _tls.token > flask_request (with try/except)
def run_sql(stmt, params=None, token=None):
    if not token:
        token = getattr(_tls, "token", None)
    if not token:
        try:
            token = flask_request.headers.get("x-forwarded-access-token")
        except RuntimeError:  # "Working outside of request context"
            token = None
```

**KEY BUG FIX**: `flask_request.headers.get(...)` RAISES `RuntimeError` in background threads. Must wrap in `try/except RuntimeError`. This applies to ALL functions that call `run_sql()` or access `flask_request` from non-request threads.

---

## 6. LLM Configuration

### Model Fallback Chain
```python
LLM_FALLBACK_CHAIN = [
    "databricks-claude-opus-4-6",
    "databricks-claude-opus-4-7",
    "databricks-claude-opus-4-5",
]
```

### _call_chat_model(prompt, token, max_tokens, model_chain=None)
- Uses Foundation Model REST API: `POST /serving-endpoints/{model}/invocations`
- NOT ai_query() (avoids SQL escaping issues)
- Per model: on 429 → wait 15s, retry once, then fall back
- On 500/502/503/timeout → immediately fall back
- On 400/401/403 → raise (non-retryable)

### _call_chat_model_race(prompt, token, max_tokens)
- Fires SAME prompt to ALL 3 endpoints simultaneously
- Returns first successful response
- Used for: single-pass analysis (≤10 tables) and merge step
- Latency = min(endpoint1, endpoint2, endpoint3)

### Parameters
| Parameter | Value |
|---|---|
| max_tokens (single pass) | 32,000 |
| max_tokens (group/merge) | 16,000 |
| API timeout | 300s per call |
| Parallelism (SQL profiling) | 20 workers |
| Parallelism (LLM groups) | 12 workers (4 × 3 endpoints) |
| Group size | 5 tables per group |

---

## 7. Profiling — Optimized for Speed

### Per-Table SQL (single query with TABLESAMPLE)
```sql
SELECT
  COUNT(*) AS __row_count__,
  COUNT(DISTINCT `col1`) AS dist__col1,
  SUM(CASE WHEN `col1` IS NULL THEN 1 ELSE 0 END) AS nulls__col1,
  CAST(MIN(`date_col`) AS STRING) AS min__date_col,  -- DATE/TIMESTAMP only
  CAST(MAX(`date_col`) AS STRING) AS max__date_col,
  (SELECT CONCAT_WS('|||', COLLECT_LIST(val))
   FROM (SELECT DISTINCT CAST(`str_col` AS STRING) AS val
         FROM table WHERE `str_col` IS NOT NULL LIMIT 3)) AS sample__str_col
FROM `catalog`.`schema`.`table` TABLESAMPLE (10 ROWS)
```

### Row Count Strategy
1. Try `DESCRIBE DETAIL table` (instant for Delta tables → `numRecords`)
2. If fails (views): estimate from sample (if sampled_rows < 10 → actual, else × 1000)

### Payload to LLM (build_profiling_payload)
- Per column: name, type, null%, cardinality_ratio
- min/max only for DATE/TIMESTAMP (LLM doesn't need INT ranges)
- 3 sample values for top 4 string columns
- Max payload: 25,000 chars

---

## 8. LLM Prompt Templates

### Single Pass (≤10 tables) — _analyze_single_pass
```
You are a senior data architect with 20 years of experience...
Analyze schema from `{catalog}.{schema}` and design OPTIMAL dimensional model.

RULES:
1. Do NOT replicate existing tables. REDESIGN them.
2. Use dim_ prefix for dimensions, fact_ for facts, bridge_ for bridges, agg_ for aggregates
3. Every FK must reference a valid PK
4. Be CONCISE in descriptions (≤15 words each)
5. Action-verb relationship descriptions

PROFILING DATA:
{payload}

Return ONLY valid JSON: { proposed_tables, relationships, columns_dropped,
  data_summary, recommended_consumers }
```

### Group Analysis (>10 tables) — _analyze_table_group
Same structure but for a subset of tables, with context about ALL table names in schema.

### Merge Step — _merge_proposals
Takes all group results, unifies into single model with cross-group relationships.

### JSON Schema (Expected LLM Response)
```json
{
  "proposed_tables": [{
    "table_name": "dim_xxx",
    "table_type": "fact|dimension|bridge|aggregate",
    "description": "...",
    "source_tables": ["original_table"],
    "columns": [{
      "name": "col_name",
      "data_type": "STRING|LONG|DOUBLE|DATE|BOOLEAN",
      "role": "pk|fk|measure|attribute",
      "source": "original_table.column or generated",
      "description": "..."
    }]
  }],
  "relationships": [{
    "from_table": "fact_xxx", "from_column": "key_col",
    "to_table": "dim_xxx", "to_column": "key_col",
    "cardinality": "1:N",
    "description_english": "Each X has many Y"
  }],
  "columns_dropped": [{"source_table": "t", "column": "c", "reason": "why"}],
  "data_summary": "3-4 sentences",
  "recommended_consumers": ["Sales Analytics", "Finance"]
}
```

---

## 9. UI Structure

### Layout (top to bottom)
1. **Header bar**: Logo (36px) + "AI Data Model Designer" title + search input + "Export Report" button
2. **Controls bar**: Catalog dropdown, Schema dropdown, Layout dropdown, "🤖 Generate Data Model" button
3. **Main area**: Cytoscape graph (left, `calc(100vh - 180px)`) + Summary panel (right, 340px)
4. **Zoom controls**: +/-/fit buttons overlaid top-right of graph

### Summary Panel Contents (right side)
- Data summary text
- Consumer badges
- Proposed tables list with type badges
- Relationships list with business descriptions
- Dropped columns (collapsible)

### Color Scheme
| Element | Color | Hex |
|---|---|---|
| Fact tables | Blue | `#2563eb` |
| Dimension tables | Green | `#16a34a` |
| Bridge tables | Orange | `#ea580c` |
| Aggregate tables | Purple | `#7c3aed` |
| PK columns | Gold | `#eab308` |
| FK columns | Blue | `#2563eb` |
| Measure columns | Green | `#16a34a` |
| Attribute columns | Gray | `#94a3b8` |

### Graph Configuration
| Setting | Value |
|---|---|
| Default layout | Cola (force-directed) |
| nodeSpacing | 80 |
| edgeLengthVal | 150 |
| maxSimulationTime | 2000ms |
| animate | **false** (instant placement — critical for fit()) |
| Node shape | round-rectangle, 200×56px |
| Edge consolidation | ONE edge per table pair (shows "N relationships") |

### Auto-Fit Callback
```javascript
// Fires at 500ms, 1.5s, 3s, 5s after elements load
function doFit() {
    var cy = el._cyreg.cy;
    cy.fit(undefined, 40);
    cy.center();
}
```
**CRITICAL**: `animate: false` on Cola layout. Without this, nodes move during 3s simulation and `fit()` catches a half-finished layout → only 1-2 nodes visible.

---

## 10. Real-Time Progress Overlay

### Architecture
```
Button click → captures token → starts daemon thread → shows modal
dcc.Interval (1.2s) → polls _job dict → updates modal steps
```

### _job Dict Structure
```python
_job = {
    "active": True/False,
    "steps": [{"time": "HH:MM:SS", "status": "active|done|error", "msg": "..."}],
    "result": {...},  # Set when complete
    "error": "...",   # Set on failure
    "cancelled": False,  # Kill switch
    "start_time": datetime
}
```

### Kill Switch
- Red "✕ Stop" button in modal → sets `_job['cancelled'] = True`
- Background thread checks at 2 points (before LLM call, before parse)
- IST timestamps (Asia/Kolkata, UTC+5:30)

---

## 11. PDF Export — Multi-Page A3 Landscape

### Architecture
- **Server-side** Python PDF generation with fpdf2
- `_generate_consulting_report(model)` produces McKinsey-style consulting report
- `_deep_sanitize()` recursively cleans all Unicode → ASCII for PDF compatibility
- `dcc.Download` component for browser download

### Page Structure
| Page | Content |
|---|---|
| **1** | Cover: logo, catalog.schema title, metric badges, full ERD graph |
| **2** | Executive summary, consumer badges, proposed tables overview table |
| **3** | Relationships table with business descriptions |
| **4+** | Table Specifications: per-table cards with column details |
| **Last** | Dropped columns with reasons |

### Table Spec Cards
- Color-coded header bar (blue=fact, green=dim, orange=bridge, purple=agg)
- Type badge + table name (dynamic width via `getTextWidth()`) + column count
- Column table: Name (48mm), Data Type (30mm), Role (20mm), Source (55mm), Description (auto)
- Role abbreviations: PK, FK, Measure, Attr (NOT full words — prevents wrapping)
- `overflow: 'ellipsize'` on all columns

### Font Sizes (minimum 9pt throughout)
| Element | Size |
|---|---|
| Page title | 28pt |
| Section headings | 15pt |
| Body text | 13-14pt |
| Table headers | 10-11pt |
| Table body | 10pt |
| Footer | 9pt |

### Error Handling (CRITICAL)
```javascript
// cy.png() can fail silently → img.onload never fires → no PDF
var png64 = null;
try { png64 = cy.png({...}); } catch(e) { }

function generatePDF(img) {
  try {
    // ... entire PDF generation ...
    if (img) { doc.addImage(img, ...); }
    else { doc.text('See interactive app view', ...); }
    doc.save(fname);
  } catch(pdfErr) {
    alert('PDF export failed: ' + pdfErr.message);
  }
}

if (png64) {
  var img = new Image();
  img.onload = function() { generatePDF(img); };
  img.onerror = function() { generatePDF(null); };
  img.src = png64;
} else {
  generatePDF(null);
}
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| REST API not ai_query() | Avoids SQL escaping, supports parallel calls, 300s timeout |
| TABLESAMPLE (10 ROWS) | 10 rows enough for cardinality ratios; avoids full scans on views |
| Explicit token passing | `threading.local()` doesn't inherit across ThreadPoolExecutor workers |
| `animate: false` on Cola | Nodes must be at final positions for `fit()` to show all nodes |
| Race mode for single calls | 3× better latency by racing all endpoints simultaneously |
| Round-robin for groups | Even load distribution prevents rate limiting on any one endpoint |
| Client-side PDF (jsPDF) | No server-side dependency; works with Cytoscape canvas export |
| `try/except RuntimeError` around flask_request | Background threads crash without this guard |
| Consolidated edges | One arrow per table pair prevents visual clutter |

---

## 13. Common Bugs & Fixes

| Bug | Cause | Fix |
|---|---|---|
| "Working outside of request context" | `flask_request` accessed in background thread | Wrap in `try/except RuntimeError` |
| "Profiled 0 tables" | Token not reaching worker threads | Pass token explicitly to `run_sql(token=...)` |
| Only 1-2 nodes visible in graph | Cola `animate: true` + early `fit()` | Set `animate: false`, fit at 500ms/1.5s/3s/5s |
| PDF export silently fails | `cy.png()` fails → `img.onload` never fires | Wrap in try/catch, use `generatePDF(null)` fallback |
| "ATTRIBUTE" wrapping in PDF | 15mm Role column too narrow | Abbreviate to "Attr", widen to 20mm |
| Text overlap in PDF tables | No overflow control | Add `overflow: 'ellipsize'` to all columns |
| JSON parse error from LLM | Response has markdown fences | Strip fences + retry logic in `_parse_llm_json()` |
| Full table scans on views | `COUNT(*)` materializes entire view | Use TABLESAMPLE only, DESCRIBE DETAIL for row count |

---

## 14. Deployment

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment
import datetime

w = WorkspaceClient()
result = w.apps.deploy(
    app_name="ai-data-model-designer",
    app_deployment=AppDeployment(
        source_code_path="/Workspace/Users/<user>/ai-data-model-designer/app"
    )
).result(timeout=datetime.timedelta(minutes=5))
print(result.status.state.value)
```

### Authorization Setup
```python
from databricks.sdk.service.apps import (
    App, AppResource, AppResourceSqlWarehouse,
    AppResourceSqlWarehouseSqlWarehousePermission,
    AppResourceServingEndpoint,
    AppResourceServingEndpointServingEndpointPermission,
)

w.apps.update(
    name="ai-data-model-designer",
    app=App(
        name="ai-data-model-designer",
        resources=[
            AppResource(name="sql-warehouse", sql_warehouse=AppResourceSqlWarehouse(
                id="<warehouse_id>",
                permission=AppResourceSqlWarehouseSqlWarehousePermission.CAN_USE)),
            AppResource(name="serving-endpoint", serving_endpoint=AppResourceServingEndpoint(
                name="databricks-claude-opus-4-5",
                permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY)),
        ],
        user_api_scopes=["sql", "serving.serving-endpoints",
                         "catalog.catalogs:read", "catalog.schemas:read", "catalog.tables:read"],
    )
)
```

---

## 15. Testing Checklist

- [ ] Select catalog/schema → tables populate
- [ ] Click "Generate Data Model" → thinking modal appears with timestamped steps
- [ ] Profiling shows "~X rows" (sampled, not full scan)
- [ ] LLM analysis completes (check logs: "Race winner: opus-4.X in Xs")
- [ ] All proposed tables visible in graph (not just 1-2)
- [ ] Click nodes → detail panel shows columns with role badges
- [ ] Summary panel shows consumers, relationships, dropped columns
- [ ] Export Report → PDF downloads with all pages
- [ ] Kill switch (✕ Stop) aborts in-progress generation
- [ ] Test on schema with 3 tables (single pass) and 20+ tables (parallel groups)

---

## 16. Performance Summary

| Stage | Technique | Impact |
|---|---|---|
| SQL Profiling | TABLESAMPLE (10 ROWS) + 20 workers | ~35× less data scanned |
| Row Count | DESCRIBE DETAIL (Delta) / estimate (views) | Eliminates full COUNT(*) |
| LLM Single Pass | Race 3 endpoints simultaneously | Latency = fastest endpoint |
| LLM Groups | Round-robin across 3 endpoints, 12 workers | 3× throughput |
| LLM Merge | Race 3 endpoints | No sequential bottleneck |
| Payload | Dropped distinct_count, DATE-only ranges | ~40% smaller prompts |
| Graph Render | Cola animate=false + 4-stage fit() | All nodes visible instantly |

---

## 17. Known Limitations

1. Views don't support DESCRIBE DETAIL — row count estimated from sample
2. PDF depends on CDN loading of jsPDF/autoTable scripts
3. 200K token context handles ~100-150 tables max
4. No DDL generation (proposes model but doesn't create tables)
5. Single-schema scope (no cross-schema relationships)
6. Claude Opus thinking time (~20-40s) is irreducible regardless of parallelism

# AI Data Model Designer

An intelligent Databricks App that analyzes your Unity Catalog schemas and proposes optimized data models — choose between **3NF Relational** or **Dimensional (Kimball Star Schema)** — using a Map-Reduce LLM pattern with multi-layer relationship detection.

## Features

- **Dual Model Types**: 3NF Relational (normalized entities) or Dimensional (Kimball star/snowflake)
- **Auto-Discovery**: Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
- **Smart Profiling**: Profiles schema data via TABLESAMPLE (10 rows per table), 20 parallel workers
- **Map-Reduce LLM**: Sonnet for fast parallel entity extraction (Map), best Opus for global relationship inference (Reduce)
- **Multi-Layer Relationship Detection**:
  - **LLM Reduce** (Opus): Global FK→PK inference across all entities
  - **Heuristic FK Detection**: Exact match (`order_id` = `order_id`) + suffix match (`origin_airport_id` ends with `_airport_id`)
  - **LLM Rescue Pass** (Sonnet): Catches semantic relationships for orphaned entities (non-standard naming like `airline_code`, `Vendor_Code`)
- **Domain-Clustered ERD**: Interactive Cytoscape graph with connected component detection, compound parent nodes, and fcose layout that groups related entities and separates unrelated domains
- **Naming Conventions**: LLM recommends best-practice names (singular nouns, snake_case, `<entity>_id` PKs)
- **PDF Export**: McKinsey-style consulting-grade multi-page A3 landscape report
- **Real-Time Progress**: Live step-by-step progress during all phases (profiling, map, reduce, validation, heuristic, rescue)
- **Cost-Efficient**: Single Opus call for Reduce (no racing), Sonnet for Map and rescue
- **Oversized Group Recovery**: Automatically splits groups that exceed context limits

## Model Types

| Mode | Entity Types | Column Roles | Use Case |
|---|---|---|---|
| **3NF Relational** | Core Entity, Weak Entity, Associative, Reference | PK, FK, Attribute | Normalized operational databases |
| **Dimensional (Star)** | Fact, Dimension, Bridge, Aggregate | PK, FK, Measure, Attribute | Analytics / data warehouse |

## Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- At least one **SQL Warehouse** (Serverless, Pro, or Classic)
- **Foundation Model API** access (Claude Sonnet + Opus endpoints)
- Permission to **create Databricks Apps** in the workspace

## Installation

### Step 1: Clone or Sync the Repository

Clone this repo into your Databricks workspace as a Git folder, or copy all files into a workspace directory.

### Step 2: Open the Installer Notebook

Open `INSTALL` notebook in the Databricks notebook UI.

### Step 3: Run Cell 1

Click **Run** on Cell 1. The installer will:

1. Auto-discover your SQL Warehouse
2. Auto-discover ALL Claude LLM endpoints (dynamic, no hardcoded names)
3. Create the Databricks App
4. Attach resources and configure permissions
5. Deploy from the `app/` subfolder
6. Print the live app URL

### Step 4 (Optional): Customize

- **Logo**: Replace `app/assets/logo.png` with your own
- **Overrides**: Uncomment Cell 2 in `INSTALL` to specify exact warehouse/endpoint IDs

## Architecture — Map-Reduce LLM Pattern

```
User selects Catalog + Schema + Model Type → clicks "Generate Data Model"
  │
  ├─ Step 1: METADATA — information_schema.columns
  ├─ Step 2: PROFILING — TABLESAMPLE (10 ROWS), 20 parallel workers
  │
  ├─ Step 3: MAP PHASE (Sonnet — fast, parallel)
  │   ├─ Per-table payloads built directly from profiles (no truncation)
  │   ├─ Groups of 3, round-robin across latest Sonnet endpoint(s)
  │   ├─ 20 parallel workers, oversized groups auto-split on 400 errors
  │   ├─ Naming conventions enforced (singular, snake_case, <entity>_id PKs)
  │   └─ Programmatic merge: Python dedup (<1ms, no LLM merge)
  │
  ├─ Step 4: REDUCE PHASE (Best Opus — single call, cost-efficient)
  │   ├─ Compressed schema catalog (~400 chars/table)
  │   ├─ Single best Opus endpoint (no racing — saves 2/3 cost)
  │   ├─ Fallback chain: next Opus → Sonnet → Haiku
  │   └─ Outputs: all FK→PK relationships
  │
  ├─ Step 5: HEURISTIC FK DETECTION (deterministic, instant)
  │   ├─ EXACT match: order.restaurant_id = restaurant.restaurant_id
  │   ├─ SUFFIX match: flight.origin_airport_id ends with airport.airport_id
  │   └─ Multi-valued PK index (defaultdict(list)) handles duplicate column names
  │
  ├─ Step 6: LLM RESCUE PASS (Sonnet — fast, cheap, targeted)
  │   ├─ Identifies orphaned entities (0 relationships after Opus + heuristic)
  │   ├─ Mini-catalog: full details for orphans, PK-only for connected entities
  │   ├─ Catches semantic matches: origin_airport→airport, airline_code→airline
  │   └─ Validates against entity schema before injection
  │
  ├─ Step 7: PARSE & VALIDATE
  │   └─ Type normalization, FK/PK integrity, role validation
  │
  └─ Step 8: RENDER — Domain-clustered ERD
      ├─ Connected component detection (BFS) → domain clusters
      ├─ Compound parent nodes per cluster ("Aviation", "Orders", etc.)
      ├─ fcose layout with dynamic scaling based on entity count
      └─ Summary panel + dynamic legend
```

### Tiered Model Strategy

| Phase | Model | Rationale |
|---|---|---|
| **Map** | Sonnet (latest version only, e.g. sonnet-4-6) | Simple structured extraction. 3-5x faster than Opus. |
| **Reduce** | Best Opus (single call, e.g. opus-4-7) | Complex cross-entity reasoning. Cost-efficient — no racing. |
| **Rescue** | Sonnet (fast, cheap) | Small targeted prompt for orphaned entities only. |

### Multi-Layer Relationship Detection

| Layer | Type | What It Catches | Example |
|---|---|---|---|
| **Opus Reduce** | LLM | Cross-entity FK→PK via reasoning | `order.customer_id → customer.customer_id` |
| **Heuristic (Exact)** | Deterministic | Column name = PK name | `payment.order_id → order.order_id` |
| **Heuristic (Suffix)** | Deterministic | Column ends with `_` + PK name | `flight.origin_airport_id → airport.airport_id` |
| **LLM Rescue** | LLM (Sonnet) | Semantic matches for non-standard naming | `flights.airline_code → airline.iata_code` |

### Endpoint Discovery & Version Sorting

The app dynamically discovers ALL Claude endpoints via the Databricks Serving API and sorts them by:
1. **Tier**: Opus (0) → Sonnet (1) → Haiku (2)
2. **Version** (descending): Versions are padded to 3 digits for correct comparison — e.g. `sonnet-4-6` → `(4,6,0)` sorts before `sonnet-4` → `(4,0,0)`, ensuring the latest sub-version is always selected.

Only the **single latest Sonnet** is used for the Map phase. Older Sonnet versions are pushed to the fallback chain.

### Throughput (200 tables)

| Config | Map | Reduce | Total |
|---|---|---|---|
| Old (all Opus, group_size=1) | ~11 min | N/A | ~11 min |
| New (Sonnet Map + Opus Reduce) | ~1.5 min | ~45s | **~2.5 min** |

## Domain-Clustered Graph Layout

The ERD uses **connected component detection** (BFS) to identify domain clusters:

1. Entities connected by relationships form a cluster (e.g., "Aviation" = flight + airline + airport + booking)
2. Each cluster gets a **compound parent node** — a dashed container with a domain label
3. **fcose layout** natively separates compound clusters with high `componentSpacing`
4. Layout params **scale dynamically** with entity count: `nodeRepulsion = max(15000, N×200)`, `componentSpacing = max(400, N×5)`

Domain naming uses keyword matching (e.g., tables containing "flight"/"airline"/"airport" → "Aviation").

## Color Scheme

### 3NF Relational
| Entity Type | Color |
|---|---|
| Core Entity | Dark Navy (#1e3a8a) |
| Weak Entity | Sky Blue (#0ea5e9), dashed border |
| Associative | Orange (#f97316) |
| Reference | Slate (#64748b), dotted border |

### Dimensional (Star)
| Entity Type | Color |
|---|---|
| Fact | Blue (#2563eb), thick border |
| Dimension | Green (#16a34a) |
| Bridge | Deep Orange (#ea580c), dashed border |
| Aggregate | Purple (#7c3aed), dotted border |

## Troubleshooting

| Issue | Solution |
|---|---|
| "No SQL Warehouse found" | Create a SQL Warehouse and rerun the installer |
| "No serving endpoints" | Enable Foundation Model APIs in your workspace |
| "Permission denied" | Ask your workspace admin for app creation privileges |
| Installer timeout (>10 min) | Check the Apps page — deployment may still be running |
| 400 BAD\_REQUEST on some groups | Oversized groups auto-split into individual calls |
| Overlapping graph nodes | Domain clustering + fcose separates unrelated entities; increase componentSpacing if needed |
| Missing relationships (payments→orders) | PK index collision fixed — uses `defaultdict(list)` for multi-valued PKs |
| Airport entity disconnected | Suffix matching catches `origin_airport_id` → `airport.airport_id` |
| Non-standard FK naming (airline\_code) | LLM rescue pass catches semantic matches for orphaned entities |
| Progress stalls after Opus responds | Fixed — 8 progress callbacks now cover parsing, validation, heuristic, rescue |
| Only N of M tables sent to LLM | Fixed: per-table payloads bypass old 500K truncation |
| Wrong Sonnet version selected | Fixed: version tuples padded to 3 digits for correct sorting |

## Project Structure

```
ai-data-model-designer-git/
├── INSTALL.ipynb               # One-click installer notebook
├── README.md                   # This file
├── SKILL.md                    # Complete build guide / skill documentation
├── app/                        # Databricks App (deployed from here)
│   ├── app.py                  # Main Dash application (~3,200 lines, ~135KB)
│   ├── app.yaml                # Databricks App manifest
│   ├── requirements.txt        # Python dependencies
│   └── assets/
│       ├── style.css           # Custom CSS (badges, radio buttons, legend)
│       ├── logo.png            # Replaceable logo
│       └── logo_data.js        # Base64 logo for PDF
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Sonnet for Map, single Opus for Reduce | Sonnet is 3-5x faster for extraction; Opus reserved for complex reasoning; no racing saves 2/3 cost |
| Multi-layer relationship detection | LLM + heuristic + rescue catches standard naming, suffix patterns, AND semantic matches |
| `defaultdict(list)` PK index | Handles schemas where multiple tables share the same PK column name (e.g., `order_id`) |
| Naming conventions in prompts | LLM recommends `<entity>_id` pattern, making suffix heuristic more effective |
| Domain-clustered fcose layout | Connected component detection groups related entities; compound nodes prevent cross-domain overlap |
| Dynamic layout scaling | `nodeRepulsion` and `componentSpacing` scale with entity count for large schemas |
| Per-table payloads (no concatenation) | Eliminates the 500K truncation bottleneck |
| Progress callbacks in Reduce | 8 `_add_step` calls prevent UI stall between Opus response and final result |
| Padded version sorting (3-digit) | Ensures `sonnet-4-6` correctly beats `sonnet-4` as "latest" |
| LLM rescue for orphans only | Cheap/fast — only processes disconnected entities, not the full 100+ table catalog |
| OverflowError on 400 → group split | Oversized groups recover by splitting into individual calls |
| Export button outside legend-bar | Prevents Dash callback conflicts when legend updates |

## License

Internal use. Modify and redistribute freely within your organization.

# AI Data Model Designer

An intelligent Databricks App that analyzes your Unity Catalog schemas and proposes optimized data models — choose between **3NF Relational** or **Dimensional (Kimball Star Schema)** — using a Map-Reduce LLM pattern.

## Features

- **Dual Model Types**: 3NF Relational (normalized entities) or Dimensional (Kimball star/snowflake)
- **Auto-Discovery**: Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
- **Smart Profiling**: Profiles schema data via TABLESAMPLE (10 rows per table), 20 parallel workers
- **Map-Reduce LLM**: Sonnet for fast parallel entity extraction (Map), Opus for global relationship inference (Reduce)
- **Professional ERD**: Interactive Cytoscape graph with color-coded entity types, shadows, and Cola force-directed layout
- **PDF Export**: McKinsey-style consulting-grade multi-page A3 landscape report
- **Real-Time Progress**: Live step-by-step progress during model generation
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
  │   └─ Programmatic merge: Python dedup (<1ms, no LLM merge)
  │
  ├─ Step 4: REDUCE PHASE (Opus — high intelligence, racing)
  │   ├─ Compressed schema catalog (~400 chars/table)
  │   ├─ 3 Opus endpoints racing simultaneously
  │   └─ Outputs: all FK→PK relationships, injected back into model
  │
  ├─ Step 5: PARSE & VALIDATE
  │   └─ Type normalization, FK/PK integrity, role validation
  │
  └─ Step 6: RENDER — Cytoscape ERD (Cola layout, randomized) + summary panel
```

### Tiered Model Strategy

| Phase | Model | Rationale |
|---|---|---|
| **Map** | Sonnet (latest version only, e.g. sonnet-4-6) | Simple structured extraction. 3-5x faster than Opus. |
| **Reduce** | Opus (3 endpoints racing, e.g. opus-4-7) | Complex cross-entity reasoning. Highest intelligence. |

### Endpoint Discovery & Version Sorting

The app dynamically discovers ALL Claude endpoints via the Databricks Serving API and sorts them by:
1. **Tier**: Opus (0) → Sonnet (1) → Haiku (2)
2. **Version** (descending): Versions are padded to 3 digits for correct comparison — e.g. `sonnet-4-6` → `(4,6,0)` sorts before `sonnet-4` → `(4,0,0)`, ensuring the latest sub-version is always selected.

Only the **single latest Sonnet** is used for the Map phase. Older Sonnet versions are pushed to the fallback chain.

### Throughput (200 tables)

| Config | Map | Reduce | Total |
|---|---|---|---|
| Old (all Opus, group_size=1) | \~11 min | N/A | \~11 min |
| New (Sonnet Map + Opus Reduce) | \~1.5 min | \~45s | **\~2.5 min** |

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
| Graph renders as vertical chain | Fixed: Cola uses `randomize: true` for proper 2D layout |
| Only N of M tables sent to LLM | Fixed: per-table payloads bypass old 500K truncation |
| Wrong Sonnet version selected | Fixed: version tuples padded to 3 digits for correct sorting |

## Project Structure

```
ai-data-model-designer-git/
├── INSTALL.ipynb               # One-click installer notebook
├── README.md                   # This file
├── SKILL.md                    # Complete build guide / skill documentation
├── app/                        # Databricks App (deployed from here)
│   ├── app.py                  # Main Dash application (~2,970 lines, ~123KB)
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
| Sonnet for Map, Opus for Reduce | Sonnet is 3-5x faster for structured extraction; Opus reserved for complex reasoning |
| Per-table payloads (no concatenation) | Eliminates the 500K truncation bottleneck |
| Padded version sorting (3-digit) | Ensures `sonnet-4-6` correctly beats `sonnet-4` as "latest" |
| Randomized Cola layout | Prevents linear chain convergence on first render |
| OverflowError on 400 → group split | Oversized groups recover by splitting into individual calls |
| Dynamic legend via callback | Updates entity type colors when switching between 3NF and Dimensional |
| Export button outside legend-bar | Prevents Dash callback conflicts when legend updates |

## License

Internal use. Modify and redistribute freely within your organization.

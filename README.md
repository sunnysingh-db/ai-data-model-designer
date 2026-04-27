# AI Data Model Designer

An intelligent Databricks App that analyzes your Unity Catalog schemas and proposes optimized star/snowflake dimensional models using LLM-powered analysis.

## Features

- **Auto-Discovery**: Reads live metadata from Unity Catalog (tables, columns, PKs, FKs)
- **Smart Profiling**: Profiles schema data via TABLESAMPLE (10 rows per table) for speed
- **LLM-Powered Design**: Distributes tables across Opus endpoints (1 table per call, programmatic merge)
- **Interactive ERD**: Color-coded Cytoscape graph with fact/dimension/bridge/aggregate tables
- **PDF Export**: McKinsey-style consulting-grade multi-page A3 landscape report
- **Real-Time Progress**: Live step-by-step progress during model generation

## Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- At least one **SQL Warehouse** (Serverless, Pro, or Classic)
- **Foundation Model API** access (Claude Opus endpoints recommended)
- Permission to **create Databricks Apps** in the workspace

## Installation

### Step 1: Clone or Sync the Repository

```
Clone this repo into your Databricks workspace as a Git folder,
or copy all files into a workspace directory.
```

### Step 2: Open the Installer Notebook

Open `INSTALL` notebook in the Databricks notebook UI.

### Step 3: Run Cell 1

Click **Run** on Cell 1. That's it. The installer will:

1. Auto-discover your SQL Warehouse
2. Auto-discover available LLM serving endpoints
3. Create the Databricks App
4. Attach resources and configure permissions
5. Deploy from the `app/` subfolder
6. Print the live app URL

```
🚀 AI Data Model Designer — Installer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[12:34:01] 🔍 Discovering SQL Warehouses...
[12:34:02] ✅ Found SQL Warehouse: "Starter Warehouse" (id: abc123)
...
[12:36:45] ✅ Deployment complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 APP IS READY!
🔗 https://your-workspace.databricks.app/ai-data-model-designer
```

### Step 4 (Optional): Customize

- **Logo**: Replace `app/assets/logo.png` with your own, then run `python app/scripts/generate_logo_b64.py`
- **Overrides**: Uncomment Cell 2 in `INSTALL` to specify exact warehouse/endpoint IDs

## Architecture

```
User selects Catalog + Schema → clicks "Generate Data Model"
  │
  ├─ Step 1: METADATA — information_schema.columns
  ├─ Step 2: PROFILING — TABLESAMPLE (10 ROWS), 20 parallel workers
  ├─ Step 3: LLM ANALYSIS — 1 table per Opus call, round-robin distribution, programmatic merge
  ├─ Step 4: VALIDATION — FK/PK integrity, orphan detection
  └─ Step 5: RENDER — Cytoscape ERD + summary panel
```

## Color Scheme

| Table Type | Color |
|---|---|
| Fact | Blue (#2563eb) |
| Dimension | Green (#16a34a) |
| Bridge | Orange (#ea580c) |
| Aggregate | Purple (#7c3aed) |

## Troubleshooting

| Issue | Solution |
|---|---|
| "No SQL Warehouse found" | Create a SQL Warehouse and rerun the installer |
| "No serving endpoints" | Enable Foundation Model APIs in your workspace |
| "Permission denied" | Ask your workspace admin for app creation privileges |
| Installer timeout (>10 min) | Check the Apps page manually — deployment may still be running |
| Only 1-2 nodes visible | This is a known Cola layout issue — the app handles it with auto-fit |

## Project Structure

```
ai-data-model-designer/
├── INSTALL.ipynb               # One-click installer notebook (root)
├── README.md                   # This file (root)
├── SKILL.md                    # Skill documentation (root)
├── app/                        # Databricks App (deployed from here)
│   ├── app.py                  # Main Dash application (~2,400 lines)
│   ├── app.yaml                # Databricks App manifest
│   ├── requirements.txt        # Python dependencies
│   ├── assets/
│   │   ├── style.css           # Custom CSS
│   │   ├── logo.png            # Replaceable logo
│   │   └── logo_data.js        # Base64 logo for PDF
│   └── scripts/
│       └── generate_logo_b64.py
└── AI Data Model Designer Implementation Plan.ipynb  # Design notes
```

## License

Internal use. Modify and redistribute freely within your organization.

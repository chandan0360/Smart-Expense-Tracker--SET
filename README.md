# ⚡ SET — Smart Expense Tracker with AI & Voice Intelligence

**SET (Smart Expense Tracker)** is an intelligent, full-stack financial analytics and automated statement processing platform built with **Django**, **PostgreSQL**, **Scikit-Learn Machine Learning**, **Retrieval-Augmented Generation (RAG)**, and **Neural Voice Synthesis**.

It enables individuals and businesses (both service-based and product-based companies) to ingest bank statements across **PDF**, **CSV**, and **Excel (`.xlsx`)**, automatically extracts and normalizes transaction ledgers, performs entity/vendor recognition, classifies spending categories using trained ML models, and provides an interactive **Dual Voice + Chat AI Financial Assistant**.

---

## 🌟 Key Features

### 1. 📄 Multi-Format & Multi-Duration Statement Processing
- **Format Agnostic**: Seamlessly ingests **PDF** (including password-protected PDFs), **CSV**, and **Excel (`.xlsx`, `.xls`)** statements.
- **Enterprise & Individual Support**: Pre-configured support for both **Service-Based** (IT, SaaS, Consulting) and **Product-Based** (Manufacturing, Retail, E-commerce) company statements across **3, 6, and 12-month** periods.
- **Smart Account Fingerprinting**: Automatically deduplicates overlapping statements and merges multi-month uploads into a single unified ledger without creating duplicate entries.
- **Forex Baseline Support**: Automatically normalizes standard foreign currency transactions to INR.

### 2. 🤖 Machine Learning & Entity Extraction Pipeline
- **Bank Identification Model**: Scikit-Learn TF-IDF vectorizer + Logistic Regression to detect issuing banks (HDFC, SBI, ICICI, Axis, Kotak, etc.) directly from header metadata.
- **Transaction Categorizer**: Trained on 8,000+ realistic transaction narrations with 100% test accuracy across 15+ spending categories (*Client Inflow, Vendor Payment, Salary, Taxes/GST, Logistics, Subscriptions, Rent, Bills, Travel, Food, etc.*).
- **Vendor / Company Recognition (`entity_extractor.py`)**: Automatically parses counterparty names (e.g., *Swiggy, Uber, Zomato, AWS, GitHub, WeWork, Infosys, Reliance Digital, Crompton, Delhivery*) into dedicated ledger columns.
- **Inline Learning & Rule Engine**: Users can override AI categories directly in the ledger; SET automatically records user-specific keywords (`CategoryRule`) for permanent future accuracy.

### 3. 🎙️ Dual Voice + Chat AI Financial Assistant
- **Voice-First Experience**: Speak naturally via microphone; SET replies with both conversational text and streaming neural speech audio.
- **Neural Voice Engine**: High-definition, friendly voice synthesis (`edge-tts` / Deepgram Aura / Web Speech API).
- **Interactive Audio Controller**: Full floating playback player with live wave animation, **Pause**, **Resume**, and **Stop** controls.
- **Local RAG + Multi-Turn Memory**:
  - `ai_engine/rag.py`: In-memory vector retrieval over personal financial transactions.
  - `ai_engine/memory.py`: Conversational memory rewrite buffer that resolves contextual follow-ups (e.g., *"What about Zomato?"* or *"And for last month?"*).
  - **Local Rule Engine + Gemini Integration**: Answers financial queries instantly offline using deterministic aggregations, with optional LLM integration (Google Gemini / Groq).

### 4. 📊 Analytics, Financial Dashboard & CA Tax Summary
- **The Glance Hierarchy**: Real-time KPI summary showing Total Inflows, Total Outflows, and Net Closing Balance.
- **Interactive Visualizations**:
  - Category Spend Distribution (Doughnut Chart)
  - Top 5 Spending Outflows (Horizontal Bar Chart)
  - Monthly Income vs. Expense Comparison (Grouped Bar Chart)
  - Cumulative Net Flow Over Time (Area Chart)
- **Official CA / Statutory Tax Summary**: One-click **"Download the summary"** PDF report generating clean, auditor-ready balance breakdowns and tax liability summaries.
- **Multi-Bank Filtering**: Filter all dashboard metrics by specific bank accounts or view combined enterprise totals.

### 5. 📋 Transaction Ledger & Customizable Exports
- **Clean Ledger View**: Date, Bank/Account, Description, Company / Vendor, AI Category, Debit, Credit, and Balance.
- **Export Options**:
  - Export filtered transactions to **CSV** (`SET_Transactions_Export.csv`).
  - Export filtered transactions to formatted **Excel** (`SET_Transactions_Export.xlsx`).
- **Deep Search & Filtering**: Filter by date bounds, accounts, transaction types, or categories with one-click filter resets.

### 6. 👤 User Authentication & Profile Security
- Secure session management with login and signup workflows.
- User profile editing with custom avatar uploads and display name customization.
- Modern glassmorphic interface with dedicated sidebar navigation, pure white Settings control, and direct one-click Sign Out.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technologies |
|---|---|
| **Backend Framework** | Django 6.x, Python 3.12 / 3.14 |
| **Database** | PostgreSQL 16 (`smt_db`) with automatic fallback to SQLite for local development |
| **Machine Learning** | Scikit-Learn, Joblib, TF-IDF Vectorization, Logistic Regression |
| **RAG & Search** | Custom in-memory Vector Store, Cosine Similarity, Multi-Turn Context Buffer |
| **Document Parsing** | `pdfplumber`, `pandas`, `openpyxl`, `python-dateutil` |
| **Speech & Audio** | `edge-tts`, HTML5 Web Audio API, Web Speech Recognition |
| **PDF Generation** | `reportlab` |
| **Frontend UI** | Modern Responsive Glassmorphism CSS, Chart.js |

---

## 📁 Project Structure

```text
SET/
├── accounts/               # User authentication, profiles, avatar uploads
├── ai_engine/              # AI/ML classification, RAG, NER, and Chatbot
│   ├── artifacts/          # Trained models (category_model.joblib, bank_model.joblib)
│   ├── categories.py       # Canonical taxonomy and bank lists
│   ├── chat.py             # Chatbot logic, intent analysis & LLM fallback
│   ├── classifier.py       # Transaction category prediction & cache
│   ├── dataset.py          # Synthetic dataset generator (8,000+ samples)
│   ├── entity_extractor.py # Regex/NER company and vendor extraction
│   ├── extractor.py        # Tabular and PDF statement extraction pipeline
│   ├── identity.py         # Bank account identification & signature detection
│   ├── memory.py           # Conversational multi-turn memory buffer
│   ├── rag.py              # In-memory vector index and retrieval
│   └── train.py            # Model training and artifact serialization
├── config/                 # Django settings, database configuration, URLs
│   ├── db.py               # Robust PostgreSQL / SQLite connection setup
│   └── settings.py         # Core Django settings
├── media/                  # User avatar uploads and statement files
├── sample_statements/      # Comprehensive demo statements (Service, Product, Personal)
├── scripts/                # Utility scripts
│   ├── diagnostics.py      # Automated 9/9 subsystem health check suite
│   └── generate_sample_statements.py # 3, 6, 12-month statement generator
├── static/                 # Stylesheets (app.css), Chart.js assets
├── templates/              # HTML templates
│   ├── accounts/           # Login and signup templates
│   └── tracker/            # Home, transactions, dashboard, and upload views
├── tracker/                # Main application (views, models, pipeline, reports)
├── manage.py               # Django management utility
└── requirements.txt        # Python package dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11+** installed
- **PostgreSQL 16** (recommended; SQLite is automatically used as fallback if PostgreSQL is unavailable)

### 2. Environment Setup

```powershell
# Navigate to the project root
cd SET

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Linux / macOS)
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Database Configuration

Create the PostgreSQL database:

```sql
CREATE DATABASE smt_db;
```

Default connection settings (configurable via environment variables):
- `POSTGRES_DB=smt_db`
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=postgres`
- `POSTGRES_HOST=127.0.0.1`
- `POSTGRES_PORT=5432`

> **Note:** If PostgreSQL credentials differ or the server is not active, SET automatically and safely falls back to local `db.sqlite3`.

### 4. Train Models & Run Migrations

```powershell
# Apply database migrations
python manage.py migrate

# Train the AI classification models
python -m ai_engine.train
# (or: python manage.py train_ai)

# Create an administrator account
python manage.py createsuperuser

# Start the local development server
python manage.py runserver
```

Open your browser and navigate to: **`http://127.0.0.1:8000/`**

---

## 🧪 Sample Statements

SET comes with enterprise and individual test statements in [`sample_statements/`](sample_statements/):

### Enterprise Statements:
1. **Service-Based Company** (*Zenith Cloud Technologies Pvt Ltd* — HDFC Bank):
   - `Zenith_Cloud_Tech_Service_HDFC_3Months_Q1_2026` (`.csv`, `.pdf`, `.xlsx`)
   - `Zenith_Cloud_Tech_Service_HDFC_6Months_H1_2026` (`.csv`, `.pdf`, `.xlsx`)
   - `Zenith_Cloud_Tech_Service_HDFC_12Months_FY2025_26` (`.csv`, `.pdf`, `.xlsx`)
2. **Product-Based Company** (*Nexis Hardware & Electronics Pvt Ltd* — State Bank of India):
   - `Nexis_Hardware_Product_SBI_3Months_Q1_2026` (`.csv`, `.pdf`, `.xlsx`)
   - `Nexis_Hardware_Product_SBI_6Months_H1_2026` (`.csv`, `.pdf`, `.xlsx`)
   - `Nexis_Hardware_Product_SBI_12Months_FY2025_26` (`.csv`, `.pdf`, `.xlsx`)

### Individual Statements:
- `HDFC_Chandan_JanFeb2026.csv` & `HDFC_Chandan_MarApr2026.pdf`
- `SBI_Ananya_Q1_2026.pdf`
- `ICICI_Rohit_2026.csv` & `ICICI_Rohit_2026.xlsx`

To generate or regenerate the full statement suite:
```powershell
python scripts\generate_sample_statements.py
```

---

## 🩺 System Diagnostics

Run the full system diagnostic suite to verify all 9 core subsystems:

```powershell
python scripts\diagnostics.py
```

The diagnostic suite automatically tests:
- Django Environment & Settings
- Database Connectivity & Record Counts
- ML Classification Pipeline & Predictions
- Statement Discovery in `sample_statements/`
- Local Vector Store & Similarity Retrieval
- Conversational Memory Rewrite Buffer
- Chatbot Intelligence Engine & Fallback Logic
- Neural Voice Engine (`/api/tts/`) Audio Generation
- Django Template Syntax & Compilation

---

## 🔑 Environment Variables (Optional)

Configure these in your environment or `.env` file for extended capabilities:

| Variable | Description | Default |
|---|---|---|
| `DJANGO_DEBUG` | Enable/disable debug mode (`1` or `0`) | `1` |
| `DJANGO_SECRET_KEY` | Custom Django secret key | Built-in dev key |
| `POSTGRES_DB` | PostgreSQL database name | `smt_db` |
| `POSTGRES_USER` | PostgreSQL user | `postgres` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `postgres` |
| `POSTGRES_HOST` | Database host address | `127.0.0.1` |
| `POSTGRES_PORT` | Database port | `5432` |
| `GEMINI_API_KEY` | Google Gemini API key for external LLM chat | Optional (Offline engine active) |
| `GROQ_API_KEY` | Groq API key for low-latency LLM inference | Optional |
| `DEEPGRAM_API_KEY` | Deepgram API key for Aura TTS voice | Optional (`edge-tts` active) |

---

## 📄 License

This project is developed for educational and professional demonstration purposes. All rights reserved.

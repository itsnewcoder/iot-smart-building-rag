# IoT Sensor Data RAG for Smart Buildings

A simple but complete Streamlit demo that combines IoT sensor streaming, document RAG (manuals/specs), anomaly detection, and predictive maintenance suggestions. Ships with a Material-like theme and ready-to-deploy on Hugging Face Spaces.

## Features
- Real-time sensor streaming (CSV simulation) with anomaly detection
- Document ingestion: PDFs/TXT of maintenance manuals and building specs
- Vector retrieval (ChromaDB) with Sentence-Transformers embeddings
- Context-aware generation via local Transformers or OpenAI (optional)
- Predictive maintenance heuristics + efficiency recommendations
- Evaluation tab (basic retrieval quality and latency)

## Quickstart

### 1) Setup
```bash
# from repo root
cd Projects/iot-smart-building-rag
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: set API keys in `.env` (at repo root):
```
OPENAI_API_KEY=your_key
```

### 2) Sample Data
- Place PDFs/TXT in `data/manuals` and `data/specs`.
- Sensor CSVs (with timestamps) in `data/sensors` (sample provided).

### 3) Run locally
```bash
streamlit run app.py
```

### 4) Deploy on Hugging Face Spaces
- Create a new Space (Streamlit)
- Push this folder contents (including `requirements.txt`, `app.py`)
- Set `OPENAI_API_KEY` secret if using OpenAI

## Project Structure
```
.
├─ app.py                     # Streamlit UI
├─ rag/
│  ├─ ingest.py               # Load & chunk documents, build vector store
│  ├─ retrieval.py            # Query vector db
│  ├─ generate.py             # LLM wrappers (local/OpenAI)
│  └─ evaluate.py             # Basic retrieval evaluation
├─ models/
│  └─ predictive.py           # Anomaly detection & maintenance heuristics
├─ data/
│  ├─ manuals/                # Add PDFs/TXT
│  ├─ specs/                  # Add PDFs/TXT
│  └─ sensors/                # CSV sensor streams
└─ .streamlit/config.toml     # Theme
```

## Notes
- Default uses `sentence-transformers/all-MiniLM-L6-v2`. Switch in `rag/ingest.py`.
- Chroma DB folder is `.chroma`. Delete to rebuild.

## License
MIT

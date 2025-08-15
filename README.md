---
title: IoT Sensor Data RAG for Smart Buildings
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.42.1"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# IoT Sensor Data RAG for Smart Buildings

## 🏢 Problem Statement

Create a RAG system that processes IoT sensor data, maintenance manuals, and building specifications to provide predictive maintenance insights and operational optimization.

## 🎯 Key Requirements

- ✅ **IoT sensor data ingestion and real-time processing**
- ✅ **Maintenance manual and building specification integration**
- ✅ **Predictive maintenance algorithm implementation**
- ✅ **Operational efficiency optimization recommendations**
- ✅ **Anomaly detection and alert systems**

## 🚀 Technical Challenges Solved

- ✅ **Real-time sensor data streaming and processing**
- ✅ **Multi-sensor data fusion and correlation**
- ✅ **Predictive modeling for equipment failure**
- ✅ **Building system integration and compatibility**
- ✅ **Energy efficiency optimization algorithms**

## 🏗️ System Architecture

### Core Components
- **RAG Engine**: Vector database (ChromaDB) with Sentence-Transformers embeddings
- **IoT Data Processor**: Real-time sensor data streaming and anomaly detection
- **Predictive Analytics**: Equipment failure prediction and maintenance recommendations
- **Document Intelligence**: PDF/TXT processing with smart chunking strategies
- **Web Interface**: Modern Streamlit dashboard with Material design theme

### Technology Stack
- **Backend**: Python, Streamlit, ChromaDB
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB with cosine similarity
- **LLM Integration**: Local Transformers + OpenAI API (optional)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly for real-time sensor monitoring

## 📊 Features

### 1. Real-Time IoT Monitoring
- Live sensor data streaming simulation
- Multi-sensor data fusion (temperature, humidity, power consumption)
- Real-time anomaly detection using rolling z-score analysis
- Interactive time-series visualizations

### 2. Intelligent Document RAG
- PDF and TXT document ingestion
- Smart text chunking (500 tokens with 50 token overlap)
- Context-aware retrieval using vector similarity
- Source attribution and relevance scoring

### 3. Predictive Maintenance
- Equipment failure prediction algorithms
- Maintenance schedule optimization
- Energy efficiency recommendations
- Anomaly-based alert systems

### 4. Evaluation & Analytics
- Retrieval accuracy metrics
- Response latency measurement
- Document relevance scoring
- System performance monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for local LLM models)
- Internet connection (for initial model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/itsnewcoder/iot-smart-building-rag.git
cd iot-smart-building-rag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory (optional):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Run Locally

```bash
streamlit run app.py
```

**Access your app at:** `http://localhost:8501`

## 📁 Project Structure

```
iot-smart-building-rag/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── rag/                       # RAG system core
│   ├── __init__.py
│   ├── ingest.py              # Document ingestion & vector store
│   ├── retrieval.py           # Context retrieval engine
│   ├── generate.py            # LLM response generation
│   └── evaluate.py            # System evaluation metrics
├── models/                     # Predictive models
│   ├── __init__.py
│   └── predictive.py          # Anomaly detection & maintenance
├── data/                       # Sample data
│   ├── manuals/               # Maintenance manuals (PDF/TXT)
│   ├── specs/                 # Building specifications
│   └── sensors/               # IoT sensor data (CSV)
└── .chroma/                   # Vector database storage
```

## 🔧 Usage Guide

### 1. Dashboard Tab
- **Start Stream**: Begin real-time sensor data simulation
- **Live Monitoring**: View real-time sensor readings and trends
- **Anomaly Detection**: See detected anomalies with z-score analysis
- **Maintenance Tips**: Get AI-powered maintenance recommendations

### 2. RAG QA Tab
- **Ask Questions**: Query maintenance procedures and building specs
- **Context Retrieval**: View relevant document chunks and sources
- **AI Responses**: Get context-aware answers from local or OpenAI models

### 3. Evaluation Tab
- **Retrieval Testing**: Test system with custom queries
- **Performance Metrics**: View latency and relevance scores
- **Quality Assessment**: Evaluate RAG system effectiveness

### 4. Data Manager Tab
- **Document Index**: View indexed documents and sources
- **File Upload**: Add new PDFs/TXTs to the knowledge base
- **Vector Store**: Manage document embeddings and storage

## 📈 Sample Queries

Try these example questions in the RAG QA tab:

- "How to reset chiller pump?"
- "What are the fault codes for HVAC systems?"
- "How to maintain building temperature sensors?"
- "What are the power consumption optimization tips?"
- "How to troubleshoot humidity sensor issues?"

## 🎯 Evaluation Metrics

### Retrieval Quality
- **Relevance Scoring**: Cosine similarity-based ranking
- **Source Attribution**: Document source tracking
- **Context Retrieval**: Top-k document retrieval

### Performance Metrics
- **Response Latency**: End-to-end query processing time
- **Throughput**: Queries processed per second
- **Memory Usage**: Vector database storage efficiency

### RAG Effectiveness
- **Context Relevance**: Retrieved document quality
- **Answer Accuracy**: Response relevance to queries
- **Source Diversity**: Multiple document source utilization

## 🌐 Deployment

### HuggingFace Spaces (Recommended)
1. Create new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as SDK
3. Upload project files
4. Set environment variables in Space settings

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Local Deployment
```bash
# Production server
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔍 Technical Implementation Details

### Embedding Strategy
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Normalization**: L2 normalization for cosine similarity
- **Chunking**: 500 tokens with 50 token overlap

### Vector Database
- **Database**: ChromaDB
- **Similarity**: Cosine distance
- **Persistence**: Local file storage (.chroma directory)
- **Indexing**: HNSW algorithm for fast retrieval

### Anomaly Detection
- **Method**: Rolling z-score analysis
- **Window Size**: 50 data points
- **Threshold**: Z-score > 3.0
- **Metrics**: Temperature, humidity, power consumption

### Predictive Maintenance
- **Algorithm**: Rule-based heuristics + statistical analysis
- **Input**: Sensor data + anomaly patterns
- **Output**: Maintenance recommendations + efficiency tips
- **Real-time**: Continuous monitoring and updates

## 🧪 Testing

### Local Testing
```bash
# Test RAG modules
python -c "from rag.ingest import ensure_vector_store; print('✅ RAG Ready')"

# Test predictive models
python -c "from models.predictive import detect_anomalies; print('✅ Models Ready')"

# Test full application
streamlit run app.py
```

### Sample Data
The system includes sample data for testing:
- **HVAC Sensor Data**: Temperature, humidity, power readings
- **Chiller Manual**: Maintenance procedures and fault codes
- **Building Specs**: System specifications and requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Use

This project was developed as part of an academic RAG system implementation course. It demonstrates:

- **RAG Architecture**: Complete retrieval-augmented generation system
- **IoT Integration**: Real-time sensor data processing
- **Predictive Analytics**: Machine learning for maintenance
- **Vector Databases**: ChromaDB implementation
- **Modern Web UI**: Streamlit-based dashboard

## 📞 Support

For questions or issues:
- **GitHub Issues**: [Create an issue](https://github.com/itsnewcoder/iot-smart-building-rag/issues)
- **Documentation**: Check this README and code comments
- **Community**: Streamlit and HuggingFace communities

## 🚀 Future Enhancements

- [ ] Real-time IoT device integration
- [ ] Advanced ML models for failure prediction
- [ ] Multi-modal document support (images, audio)
- [ ] API endpoints for external systems
- [ ] Mobile-responsive interface
- [ ] Advanced analytics dashboard
- [ ] Integration with building management systems

---

**Built with ❤️ for Smart Building Intelligence**

*Last updated: January 2025*

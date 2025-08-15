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

## 🏢 What I Built

I created a complete RAG (Retrieval-Augmented Generation) system for smart buildings that:
- Processes real-time IoT sensor data (temperature, humidity, power)
- Integrates maintenance manuals and building specifications
- Provides predictive maintenance insights
- Detects anomalies in sensor data
- Gives operational optimization recommendations

## 🎯 Key Features

- **Real-time IoT Monitoring**: Live sensor data streaming with anomaly detection
- **Smart Document Search**: Ask questions about maintenance and get AI-powered answers
- **Predictive Analytics**: Equipment failure prediction and maintenance tips
- **Modern Dashboard**: Clean Streamlit interface with real-time visualizations

## 🚀 How to Run Locally

### Step 1: Clone and Setup
```bash
git clone https://github.com/itsnewcoder/iot-smart-building-rag.git
cd iot-smart-building-rag
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```

**Your app will open at:** `http://localhost:8501`

## 🌐 Live Demo

**Try it online:** [https://huggingface.co/spaces/imnikhilraj/iot-smart-building-rag](https://huggingface.co/spaces/imnikhilraj/iot-smart-building-rag)

## 🔧 How to Use

### Dashboard Tab
1. Click "Start Stream" to begin sensor data simulation
2. View real-time temperature, humidity, and power readings
3. See detected anomalies and maintenance recommendations

### RAG QA Tab
1. Ask questions like "How to reset chiller pump?"
2. Get AI-powered answers based on maintenance manuals
3. View source documents and relevance scores

### Evaluation Tab
1. Test the system with custom queries
2. See retrieval performance metrics
3. Check response latency and accuracy

### Data Manager Tab
1. View indexed documents
2. Upload new PDFs/TXTs to expand knowledge base

## 📁 Project Structure

```
iot-smart-building-rag/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python packages
├── rag/                       # RAG system core
│   ├── ingest.py              # Document processing
│   ├── retrieval.py           # Search engine
│   ├── generate.py            # AI responses
│   └── evaluate.py            # Performance metrics
├── models/                     # Predictive models
│   └── predictive.py          # Anomaly detection
├── data/                       # Sample data
│   ├── manuals/               # Maintenance guides
│   ├── specs/                 # Building specs
│   └── sensors/               # IoT sensor data
└── .streamlit/                 # App configuration
```

## 🧪 Sample Data Included

- **HVAC Sensor Data**: Temperature, humidity, power consumption readings
- **Chiller Manual**: Maintenance procedures and fault codes
- **Building Specifications**: System requirements and configurations

## 🔍 Technical Details

- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB with cosine similarity
- **LLM**: Local Transformers + OpenAI API (optional)
- **Anomaly Detection**: Rolling z-score analysis
- **Chunking**: 500 tokens with 50 token overlap

## 📊 What I Achieved

✅ **IoT Data Processing**: Real-time sensor streaming and analysis
✅ **Document RAG**: Intelligent search through manuals and specs
✅ **Predictive Maintenance**: Equipment failure prediction algorithms
✅ **Anomaly Detection**: Statistical analysis for sensor anomalies
✅ **Modern UI**: Professional Streamlit dashboard
✅ **Evaluation Metrics**: Performance testing and quality assessment

## 🎓 Academic Project

This project demonstrates:
- Complete RAG system implementation
- IoT data integration and processing
- Predictive analytics for smart buildings
- Vector database usage (ChromaDB)
- Modern web application development

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Server
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📞 Support

- **GitHub Issues**: [Report bugs here](https://github.com/itsnewcoder/iot-smart-building-rag/issues)
- **Live Demo**: [https://huggingface.co/spaces/imnikhilraj/iot-smart-building-rag](https://huggingface.co/spaces/imnikhilraj/iot-smart-building-rag)
- **Source Code**: [https://github.com/itsnewcoder/iot-smart-building-rag](https://github.com/itsnewcoder/iot-smart-building-rag)

## 📄 License

MIT License - feel free to use and modify!

---

**Built by Nikhil Raj** 🚀

*Last updated: January 2025*

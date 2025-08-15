import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from rag.ingest import ensure_vector_store, list_documents
from rag.retrieval import retrieve_context
from rag.generate import generate_answer
from models.predictive import (
	start_stream_simulation,
	stop_stream_simulation,
	get_latest_stream_frame,
	detect_anomalies,
	maintenance_recommendations,
	efficiency_tips,
)

st.set_page_config(page_title="Smart Building RAG", page_icon="🏢", layout="wide")

with st.sidebar:
	st.header("Configuration")
	embedding_model = st.selectbox(
		"Embedding model", ["sentence-transformers/all-MiniLM-L6-v2"], index=0
	)
	llm_backend = st.selectbox("Generator", ["Local (Transformer)", "OpenAI"], index=0)
	openai_api_key = st.text_input("OpenAI API Key", type="password") if llm_backend == "OpenAI" else ""
	rebuild = st.button("Rebuild Vector Store")

index_path = Path(".chroma")
if rebuild or not index_path.exists():
	with st.spinner("Building vector store..."):
		ensure_vector_store(embedding_model)
		st.success("Vector store ready.")

main_tabs = st.tabs(["Dashboard", "RAG QA", "Evaluation", "Data Manager"]) 

with main_tabs[0]:
	st.subheader("IoT Streaming & Anomaly Detection")
	left, right = st.columns([2, 1])

	with left:
		if st.button("Start Stream", key="start_stream"):
			start_stream_simulation("data/sensors", interval_seconds=1.0)
		if st.button("Stop Stream", key="stop_stream"):
			stop_stream_simulation()

		frame = get_latest_stream_frame()
		if frame is not None and not frame.empty:
			st.caption(f"Last update: {pd.to_datetime(frame['timestamp'].iloc[-1])}")
			fig = px.line(frame, x="timestamp", y=[c for c in frame.columns if c not in ["timestamp", "sensor_id"]], title="Live Sensors")
			st.plotly_chart(fig, use_container_width=True)
		else:
			st.info("No streaming data yet. Click 'Start Stream' to simulate.")

	with right:
		st.markdown("**Anomalies (rolling z-score > 3)**")
		frame = get_latest_stream_frame()
		if frame is not None and not frame.empty:
			anom = detect_anomalies(frame)
			st.dataframe(anom.tail(20), use_container_width=True)
			st.markdown("**Maintenance Recommendations**")
			recs = maintenance_recommendations(frame, anom)
			for r in recs:
				st.write("- ", r)
			st.markdown("**Efficiency Tips**")
			for t in efficiency_tips(frame):
				st.write("- ", t)
		else:
			st.info("Waiting for sensor data...")

with main_tabs[1]:
	st.subheader("RAG over Manuals & Building Specs")
	query = st.text_input("Ask a question (e.g., 'How to reset chiller pump?')")
	if st.button("Retrieve & Answer") and query.strip():
		with st.spinner("Retrieving context..."):
			docs = retrieve_context(query, k=4)
			answer = generate_answer(query, docs, llm_backend=llm_backend, openai_api_key=openai_api_key)
		st.markdown("**Answer**")
		st.write(answer)
		st.markdown("**Sources**")
		for d in docs:
			st.caption(f"{d.get('source','?')} | score={d.get('score','-')}")

with main_tabs[2]:
	from rag.evaluate import quick_retrieval_eval
	st.subheader("Basic Retrieval Evaluation")
	test_queries = st.text_area("Test queries (one per line)", "chiller maintenance\naf sensor fault codes\nAHU filter replacement")
	if st.button("Run Eval"):
		queries = [q.strip() for q in test_queries.splitlines() if q.strip()]
		with st.spinner("Evaluating..."):
			results = quick_retrieval_eval(queries)
		st.json(results)

with main_tabs[3]:
	st.subheader("Documents & Index")
	st.write("Indexed documents:")
	st.dataframe(pd.DataFrame(list_documents()), use_container_width=True)
	uploaded = st.file_uploader("Upload new manual/spec (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
	if uploaded:
		for f in uploaded:
			out = Path("data/manuals" if f.name.lower().endswith(".pdf") else "data/specs") / f.name
			with open(out, "wb") as w:
				w.write(f.getbuffer())
		st.success("Files saved. Click 'Rebuild Vector Store' in sidebar.")

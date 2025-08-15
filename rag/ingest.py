import os
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

CHROMA_DIR = ".chroma"
COLLECTION = "building_docs"


def _load_text_from_pdf(path: Path) -> str:
	reader = PdfReader(str(path))
	pages = []
	for p in reader.pages:
		try:
			pages.append(p.extract_text() or "")
		except Exception:
			pages.append("")
	return "\n".join(pages)


def _load_text_from_txt(path: Path) -> str:
	return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
	words = text.split()
	chunks = []
	start = 0
	while start < len(words):
		end = min(len(words), start + max_tokens)
		chunks.append(" ".join(words[start:end]))
		start = end - overlap
		if start < 0:
			start = 0
		if end == len(words):
			break
	return chunks


def _get_client():
	client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
	return client


def ensure_vector_store(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
	client = _get_client()
	if COLLECTION in [c.name for c in client.list_collections()]:
		client.delete_collection(COLLECTION)
	collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

	encoder = SentenceTransformer(embedding_model)

	docs: List[Dict] = []
	for folder in [Path("data/manuals"), Path("data/specs")]:
		folder.mkdir(parents=True, exist_ok=True)
		for p in folder.glob("**/*"):
			if p.suffix.lower() not in [".pdf", ".txt"]:
				continue
			text = _load_text_from_pdf(p) if p.suffix.lower() == ".pdf" else _load_text_from_txt(p)
			chunks = _chunk_text(text)
			for idx, ch in enumerate(chunks):
				docs.append({
					"id": f"{p.name}-{idx}",
					"text": ch,
					"source": str(p)
				})

	if not docs:
		# Seed with minimal content to avoid empty index
		docs = [{"id": "seed-0", "text": "No documents yet. Add PDFs or TXTs to data folders.", "source": "seed"}]

	embeddings = encoder.encode([d["text"] for d in docs], normalize_embeddings=True).tolist()
	collection.add(ids=[d["id"] for d in docs], embeddings=embeddings, documents=[d["text"] for d in docs], metadatas=[{"source": d["source"]} for d in docs])
	# ChromaDB automatically persists when persist_directory is set


def list_documents() -> List[Dict]:
	client = _get_client()
	if COLLECTION not in [c.name for c in client.list_collections()]:
		return []
	coll = client.get_collection(COLLECTION)
	# Return unique sources
	sources = set()
	rows = []
	# Chroma does not support listing directly; we stored metadata per item
	# We sample via export API
	export = coll.get(include=["metadatas"], ids=None, where=None, limit=10000)
	for md in export.get("metadatas", []):
		src = md.get("source", "?")
		if src not in sources:
			sources.add(src)
			rows.append({"source": src})
	return rows

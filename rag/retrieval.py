from typing import List, Dict
import chromadb
from chromadb.config import Settings

CHROMA_DIR = ".chroma"
COLLECTION = "building_docs"


def retrieve_context(query: str, k: int = 4) -> List[Dict]:
	client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
	coll = client.get_collection(COLLECTION)
	res = coll.query(query_texts=[query], n_results=k, include=["metadatas", "distances", "documents"])
	docs = []
	for i in range(len(res["ids"][0])):
		docs.append({
			"text": res["documents"][0][i],
			"source": res["metadatas"][0][i].get("source", "?"),
			"score": 1.0 - res["distances"][0][i] if res.get("distances") else None,
		})
	return docs

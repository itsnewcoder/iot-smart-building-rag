from typing import List, Dict
import time
from .retrieval import retrieve_context


def quick_retrieval_eval(queries: List[str]) -> Dict:
	results = []
	latencies = []
	for q in queries:
		start = time.time()
		docs = retrieve_context(q, k=4)
		lat = time.time() - start
		latencies.append(lat)
		results.append({
			"query": q,
			"top_sources": [d.get("source", "?") for d in docs],
			"top_scores": [d.get("score") for d in docs],
			"latency_s": round(lat, 3),
		})
	return {
		"queries": results,
		"avg_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
	}

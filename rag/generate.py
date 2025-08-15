from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

try:
	from openai import OpenAI
except Exception:
	OpenAI = None

_LOCAL_MODEL = "sshleifer/tiny-gpt2"
_tokenizer = None
_model = None


def _ensure_local():
	global _tokenizer, _model
	if _tokenizer is None or _model is None:
		_tokenizer = AutoTokenizer.from_pretrained(_LOCAL_MODEL)
		_model = AutoModelForCausalLM.from_pretrained(_LOCAL_MODEL)
		_model.to("cpu")


def _format_prompt(query: str, docs: List[Dict]) -> str:
	sources = "\n\n".join([f"[Source: {d.get('source','?')}]\n{d.get('text','')}" for d in docs])
	return (
		"You are a maintenance assistant for smart buildings. Answer briefly based on the context.\n\n"
		f"Question: {query}\n\n"
		f"Context:\n{sources}\n\n"
		"Answer:"
	)


def generate_answer(query: str, docs: List[Dict], llm_backend: str = "Local (Transformer)", openai_api_key: str = "") -> str:
	prompt = _format_prompt(query, docs)
	if llm_backend == "OpenAI" and openai_api_key and OpenAI is not None:
		client = OpenAI(api_key=openai_api_key)
		resp = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[{"role": "system", "content": "You are a helpful building maintenance assistant."}, {"role": "user", "content": prompt}],
			temperature=0.2,
		)
		return resp.choices[0].message.content

	# Local fallback
	_ensure_local()
	inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
	with torch.no_grad():
		out = _model.generate(**inputs, max_new_tokens=120, do_sample=False)
	text = _tokenizer.decode(out[0], skip_special_tokens=True)
	return text.split("Answer:")[-1].strip()

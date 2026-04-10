import os
import json
from typing import Any, Dict

import requests

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_instance = None

# 1. Configure the 4-bit quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class TransformersLLM:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2. Load the model with the quantization config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, # This is the key change!
            device_map="auto",             # Automatically uses your RTX 3060
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # (Rest of your generation code remains the same)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class TokenizerShim:
    """Minimal tokenizer-like shim exposing apply_chat_template used by the codebase.

    It implements a very small apply_chat_template to convert message arrays into
    a plain text prompt. This keeps other components working when using a remote
    LLM server that doesn't provide a tokenizer object.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        lines = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            lines.append(f"[{role}] {content}")
        prompt = "\n".join(lines)
        if add_generation_prompt:
            prompt = prompt + "\n[assistant]"
        return prompt


class HTTPLLM:
    """Generic HTTP adapter for local LLM servers (ollama, llamacpp, etc.).

    Configuration:
    - set `LLM_SERVER_URL` to the server endpoint to POST generation requests to.
      The adapter will POST JSON: {"prompt": ..., "max_new_tokens": N, "model": optional}
    - set `LLM_BACKEND_MODEL` to pass a model name (useful for ollama).

    The server is expected to return JSON with a `text` field containing the
    generated continuation. If your local server uses a different contract,
    update this adapter accordingly.
    """

    def __init__(self, server_url: str, model: str | None = None):
        self.server_url = server_url
        self.model = model
        self.tokenizer = TokenizerShim()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        payload: Dict[str, Any] = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": False}
        if self.model:
            payload["model"] = self.model

        try:
            resp = requests.post(self.server_url, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"LLM HTTP request failed: {e}")

        # try to parse common response shapes
        try:
            j = resp.json()
            return j.get('response')

        except Exception:
            # fallback to raw text
            return resp.text.strip()



def get_llm():
    """Return the singleton LLM instance, loading it on first call.

    Use env vars to select backend:
    - `LLM_BACKEND`: 'transformers' (default) or 'http'
    - `LLM_SERVER_URL`: required when `LLM_BACKEND=http`
    - `LLM_BACKEND_MODEL`: optional model name for HTTP backend (e.g., ollama model)
    """
    global _instance
    if _instance is not None:
        return _instance

    backend = os.environ.get("LLM_BACKEND", "transformers").lower()
    if backend == "transformers":
        _instance = TransformersLLM()
        return _instance

    if backend == "http":
        server_url = os.environ.get("LLM_SERVER_URL")
        if not server_url:
            raise RuntimeError("LLM_BACKEND=http requires LLM_SERVER_URL to be set")
        model = os.environ.get("LLM_BACKEND_MODEL")
        _instance = HTTPLLM(server_url=server_url, model=model)
        return _instance

    raise RuntimeError(f"Unknown LLM_BACKEND: {backend}")

if __name__ == "__main__":
    llm = get_llm()
    response = llm.generate("What is the capital of France?")
    print(response)

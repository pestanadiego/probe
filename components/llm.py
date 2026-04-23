import os
import json
from typing import Any, Dict
from openai import OpenAI

import requests

import torch
# BitsAndBytesConfig removed: Not compatible with Apple Silicon
from transformers import AutoModelForCausalLM, AutoTokenizer

# Updated to use the GGUF quantized model and base tokenizer
MODEL_NAME = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GGUF_FILE = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"

_instance = None

class TransformersLLM:
    def __init__(self, model_name: str = MODEL_NAME):
        # 1. Load the tokenizer from the official Meta repository
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # 2. Load the 4-bit GGUF model optimized for Apple Silicon
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=GGUF_FILE,
            device_map="mps",  # Maps directly to Apple's Metal Performance Shaders
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        # Llama 3.1 requires a specific chat template. 
        # If the prompt isn't formatted this way, the model will output gibberish.
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Tokenize and explicitly send to the Apple Silicon GPU ("mps")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Slice off the input prompt so the model doesn't repeat your question back to you
        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

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


class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.tokenizer = TokenizerShim()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"OpenRouter request failed: {e}")
        return response.choices[0].message.content.strip()


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
    print("BACKEND: ", backend)
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

    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_BACKEND=openrouter requires OPENROUTER_API_KEY to be set")
        model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        _instance = OpenRouterLLM(api_key=api_key, model=model)
        return _instance

    raise RuntimeError(f"Unknown LLM_BACKEND: {backend}")

if __name__ == "__main__":
    llm = get_llm()
    response = llm.generate("Explain the advantages of Apple's unified memory architecture.")
    print(response)
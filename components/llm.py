import os
import json
from typing import Any, Dict
from openai import OpenAI

import requests

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GGUF_FILE = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"

_instance = None

class TransformersLLM:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME) # load tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=GGUF_FILE,
            device_map="mps",
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

class TokenizerShim:
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

# HTTP adapter for local LLM
class HTTPLLM:
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

        try:
            j = resp.json()
            return j.get('response')

        except Exception:
            return resp.text.strip()

# Adapter for OpenRouter
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

# LLM instance
def get_llm():
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
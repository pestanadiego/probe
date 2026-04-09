import os

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class DenseRetriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.index = None

    def _encode(self, texts, batch_size=64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)
            # mean pooling over non-padding tokens
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            # l2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def build_index(self, texts):
        self.index = self._encode(texts)

    def save_index(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.index, path)

    def load_index(self, path):
        self.index = torch.load(path, weights_only=True)

    def search(self, query, top_k=20):
        query_emb = self._encode([query])
        scores = torch.matmul(query_emb, self.index.T).squeeze(0) # matrix multiplication
        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, top_k)
        return top_indices.tolist(), top_scores.tolist()

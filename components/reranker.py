import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from components.memory import Chunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return [], []

        pairs = [[query, c.text] for c in chunks]
        encoded = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
        scores = torch.sigmoid(logits).tolist()

        # sort by score descending, take top_k
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        reranked_chunks = []
        reranked_scores = []
        for chunk, score in scored:
            reranked_chunks.append(Chunk(
                text=chunk.text,
                source=chunk.source,
                score=score,
            ))
            reranked_scores.append(score)

        return reranked_chunks, reranked_scores

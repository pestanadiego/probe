import os

from components.bm25 import BM25, tokenize
from components.chunker import chunk_directory
from components.dense import DenseRetriever
from components.memory import Chunk

def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever:
    def __init__(self, dense_retriever):
        self.dense = dense_retriever
        self.bm25 = None
        self.chunks = []

    def build_index(self, corpus_dir, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        embeddings_path = os.path.join(index_dir, "embeddings.pt")

        self.chunks = chunk_directory(corpus_dir)
        texts = [c.text for c in self.chunks]
        corpus_tokens = [tokenize(t) for t in texts]

        # bm25
        if os.path.exists(bm25_path):
            self.bm25 = BM25.load(bm25_path)
        else:
            self.bm25 = BM25(corpus_tokens)
            self.bm25.save(bm25_path)

        # dense
        if os.path.exists(embeddings_path):
            self.dense.load_index(embeddings_path)
        else:
            self.dense.build_index(texts)
            self.dense.save_index(embeddings_path)

    def load_index(self, index_dir, corpus_dir):
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        embeddings_path = os.path.join(index_dir, "embeddings.pt")

        self.chunks = chunk_directory(corpus_dir)
        self.bm25 = BM25.load(bm25_path)
        self.dense.load_index(embeddings_path)

    def search(self, query, top_k=20):
        # bm25 ranking
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.scores(query_tokens)
        bm25_ranked = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]

        # dense ranking
        dense_indices, _ = self.dense.search(query, top_k)

        fused = reciprocal_rank_fusion([bm25_ranked, dense_indices])
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]

        # build result chunks with reciprocal rank fusion scores
        results = []
        rrf_scores = {doc_id: score for doc_id, score in fused}
        for doc_id in top_ids:
            chunk = self.chunks[doc_id]
            results.append(Chunk(
                text=chunk.text,
                source=chunk.source,
                score=rrf_scores[doc_id],
            ))
        return results

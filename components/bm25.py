import math
import os
import pickle

K1 = 1.5
B = 0.75

def tokenize(text):
    return text.lower().split()

class BM25:
    def __init__(self, corpus_tokens=None):
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_dl = 0.0
        self.n_docs = 0
        self.idf = {}
        self.index = {}  # token -> [(doc_id, tf), ...]

        if corpus_tokens is not None:
            self._build(corpus_tokens)

    def _build(self, corpus_tokens):
        self.n_docs = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avg_dl = sum(self.doc_lens) / self.n_docs if self.n_docs else 1.0

        for doc_id, doc in enumerate(corpus_tokens):
            term_counts = {}
            for token in doc:
                term_counts[token] = term_counts.get(token, 0) + 1
            for token, tf in term_counts.items():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append((doc_id, tf))

        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query_tokens):
        result = [0.0] * self.n_docs
        for q in query_tokens:
            if q not in self.index:
                continue
            q_idf = self.idf[q]
            for doc_id, tf in self.index[q]:
                dl = self.doc_lens[doc_id]
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / self.avg_dl)
                result[doc_id] += q_idf * numerator / denominator
        return result

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

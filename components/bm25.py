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
        self.corpus_tokens = []

        if corpus_tokens is not None:
            self._build(corpus_tokens)

    def _build(self, corpus_tokens):
        self.corpus_tokens = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avg_dl = sum(self.doc_lens) / self.n_docs if self.n_docs else 1.0

        # compute document frequencies
        for doc in corpus_tokens:
            seen = set(doc)
            for token in seen:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # compute idf
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query_tokens):
        result = [0.0] * self.n_docs
        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for i, doc in enumerate(self.corpus_tokens):
                tf = doc.count(q)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / self.avg_dl)
                result[i] += q_idf * numerator / denominator
        return result

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

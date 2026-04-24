import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.agent import DEMO_CORPUS_DIR, DEMO_INDEX_DIR
from components.dense import DenseRetriever
from components.retriever import HybridRetriever

def main():
    retriever = HybridRetriever(DenseRetriever())
    retriever.build_index(corpus_dir=DEMO_CORPUS_DIR, index_dir=DEMO_INDEX_DIR)
    print(f"index built at {DEMO_INDEX_DIR}")

if __name__ == "__main__":
    main()

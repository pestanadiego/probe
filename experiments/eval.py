import json
import os
import random
import sys
import tempfile
import time

from tqdm import tqdm

# allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from components.agent import run_with_components
from components.bm25 import BM25, tokenize
from components.dense import DenseRetriever
from components.memory import Chunk
from components.reranker import Reranker
from components.retriever import HybridRetriever, reciprocal_rank_fusion
from experiments.metrics import exact_match, token_f1

RANDOM_SEED = 42
N_SAMPLE = 250
TOP_K_RETRIEVAL = 20
MRR_K = 10

_start_time = time.time()

TECHQA_DEV_PATH = "corpus/eval/techqa/TechQA/training_and_dev/dev_Q_A.json"
TECHQA_NOTES_PATH = "corpus/eval/techqa/TechQA/training_and_dev/training_dev_technotes.json"
RESULTS_PATH = "results/eval_results.json"


def load_techqa_dev(dev_path: str) -> list[dict]:
    with open(dev_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_technotes(notes_path: str) -> dict[str, dict]:
    with open(notes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_answerable(questions: list[dict], n: int, seed: int) -> list[dict]:
    answerable = [q for q in questions if q.get("ANSWERABLE") == "Y"]
    random.seed(seed)
    return random.sample(answerable, min(n, len(answerable)))


def _log(msg: str):
    elapsed = time.time() - _start_time
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


def write_technotes_to_dir(doc_ids: list[str], technotes: dict[str, dict], directory: str):
    # write each technote as a plain-text file: {doc_id}.txt
    missing = 0
    for doc_id in tqdm(doc_ids, desc="  writing technote files", leave=False):
        note = technotes.get(doc_id)
        if note is None:
            missing += 1
            continue
        title = note.get("title", "")
        text = note.get("text", "")
        content = f"{title}\n\n{text}" if title else text
        file_path = os.path.join(directory, f"{doc_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    if missing:
        _log(f"  warning: {missing} doc ids not found in technotes corpus")


def compute_mrr_single(
    ranked_sources: list[str],
    gold_doc_id: str,
    k: int = MRR_K,
) -> float:
    # ranked_sources is a list of source filenames in rank order
    target = f"{gold_doc_id}.txt"
    for rank, source in enumerate(ranked_sources[:k], start=1):
        if source == target:
            return 1.0 / rank
    return 0.0


def bm25_only_ranking(
    query: str,
    bm25: BM25,
    chunks: list[Chunk],
    top_k: int = TOP_K_RETRIEVAL,
) -> list[str]:
    scores = bm25.scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i].source for i in ranked_indices]


def dense_only_ranking(
    query: str,
    dense: DenseRetriever,
    chunks: list[Chunk],
    top_k: int = TOP_K_RETRIEVAL,
) -> list[str]:
    indices, _ = dense.search(query, top_k)
    return [chunks[i].source for i in indices]


def hybrid_ranking(
    query: str,
    retriever: HybridRetriever,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[str]:
    result_chunks = retriever.search(query, top_k)
    return [c.source for c in result_chunks]


def run_evaluation():
    global _start_time
    _start_time = time.time()

    _log("loading TechQA dev set...")
    questions = load_techqa_dev(TECHQA_DEV_PATH)
    technotes = load_technotes(TECHQA_NOTES_PATH)
    _log(f"loaded {len(questions)} questions, {len(technotes)} technotes")

    sampled = sample_answerable(questions, N_SAMPLE, RANDOM_SEED)
    _log(f"sampled {len(sampled)} answerable questions")

    # collect all unique doc ids referenced by sampled questions
    doc_ids: set[str] = set()
    for q in sampled:
        doc_ids.update(q.get("DOC_IDS", []))
    # always include the gold document
    for q in sampled:
        if q.get("DOCUMENT") and q["DOCUMENT"] != "-":
            doc_ids.add(q["DOCUMENT"])
    _log(f"corpus: {len(doc_ids)} unique documents across sampled questions")

    with tempfile.TemporaryDirectory() as tmpdir:
        _log("writing technotes to temp dir...")
        write_technotes_to_dir(list(doc_ids), technotes, tmpdir)
        _log("technotes written")

        _log("loading dense retriever model (sentence-transformers)...")
        dense = DenseRetriever()
        _log("dense model loaded")

        _log("building retrieval index (chunking + BM25 + dense encoding) — this may take a few minutes...")
        retriever = HybridRetriever(dense)
        retriever.build_index(corpus_dir=tmpdir, index_dir=os.path.join(tmpdir, "idx"))
        n_chunks = len(retriever.chunks)
        _log(f"index built: {n_chunks} chunks")

        _log("loading reranker model (cross-encoder)...")
        reranker = Reranker()
        _log("reranker loaded — starting evaluation loop")

        results = []
        running_em = 0.0
        running_f1 = 0.0

        pbar = tqdm(sampled, desc="evaluating", unit="q")
        for i, q in enumerate(pbar, start=1):
            question_text = q.get("QUESTION_TITLE", "") + " " + q.get("QUESTION_TEXT", "")
            question_text = question_text.strip()
            gold_answer = q.get("ANSWER", "")
            gold_doc_id = q.get("DOCUMENT", "")

            pbar.set_postfix_str(f"q{i}: retrieving...")

            # per-retriever mrr using the question as the query proxy
            bm25_sources = bm25_only_ranking(question_text, retriever.bm25, retriever.chunks)
            dense_sources = dense_only_ranking(question_text, retriever.dense, retriever.chunks)
            hybrid_sources = hybrid_ranking(question_text, retriever)

            mrr_bm25 = compute_mrr_single(bm25_sources, gold_doc_id)
            mrr_dense = compute_mrr_single(dense_sources, gold_doc_id)
            mrr_hybrid = compute_mrr_single(hybrid_sources, gold_doc_id)

            pbar.set_postfix_str(f"q{i}: running agent...")

            # full agent run
            trace = run_with_components(question_text, retriever, reranker)
            predicted_answer = trace.final_answer
            n_iterations = len(trace.iterations)

            em = exact_match(predicted_answer, gold_answer)
            f1 = token_f1(predicted_answer, gold_answer)

            running_em += int(em)
            running_f1 += f1

            pbar.set_postfix_str(
                f"em={running_em/i:.3f} f1={running_f1/i:.3f} iters={n_iterations}"
            )

            results.append({
                "question_id": q.get("QUESTION_ID", ""),
                "question": question_text,
                "gold_answer": gold_answer,
                "gold_doc_id": gold_doc_id,
                "predicted_answer": predicted_answer,
                "exact_match": int(em),
                "token_f1": round(f1, 4),
                "mrr_bm25": round(mrr_bm25, 4),
                "mrr_dense": round(mrr_dense, 4),
                "mrr_hybrid": round(mrr_hybrid, 4),
                "n_iterations": n_iterations,
            })

    # compute summary statistics
    mean_em = sum(r["exact_match"] for r in results) / len(results)
    mean_f1 = sum(r["token_f1"] for r in results) / len(results)
    mean_mrr_bm25 = sum(r["mrr_bm25"] for r in results) / len(results)
    mean_mrr_dense = sum(r["mrr_dense"] for r in results) / len(results)
    mean_mrr_hybrid = sum(r["mrr_hybrid"] for r in results) / len(results)

    # f1 by iteration bucket (1, 2, 3, 4, 5, 6)
    f1_by_iter: dict[int, list[float]] = {}
    for r in results:
        n = r["n_iterations"]
        f1_by_iter.setdefault(n, []).append(r["token_f1"])
    mean_f1_by_iter = {
        str(k): round(sum(v) / len(v), 4)
        for k, v in sorted(f1_by_iter.items())
    }

    output = {
        "summary": {
            "n_questions": len(results),
            "mean_exact_match": round(mean_em, 4),
            "mean_token_f1": round(mean_f1, 4),
            "mrr_at_10_bm25": round(mean_mrr_bm25, 4),
            "mrr_at_10_dense": round(mean_mrr_dense, 4),
            "mrr_at_10_hybrid": round(mean_mrr_hybrid, 4),
            "mean_f1_by_iterations": mean_f1_by_iter,
        },
        "questions": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    total_time = time.time() - _start_time
    _log(f"done — results saved to {RESULTS_PATH}  (total {total_time/60:.1f} min)")
    print(f"\nexact match:    {mean_em:.4f}")
    print(f"token f1:       {mean_f1:.4f}")
    print(f"mrr@10 bm25:    {mean_mrr_bm25:.4f}")
    print(f"mrr@10 dense:   {mean_mrr_dense:.4f}")
    print(f"mrr@10 hybrid:  {mean_mrr_hybrid:.4f}")


if __name__ == "__main__":
    run_evaluation()

import json
import os
import sys

RESULTS_PATH = "results/eval_results.json"

# bar width for ASCII bar charts
BAR_WIDTH = 40


def _ascii_bar(value: float, max_value: float = 1.0) -> str:
    filled = int(round(value / max_value * BAR_WIDTH))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {value:.4f}"


def print_mrr_comparison(summary: dict):
    print("=" * 60)
    print("PLOT 1 — MRR@10: BM25 vs Dense vs Hybrid")
    print("=" * 60)
    metrics = {
        "BM25 alone": summary["mrr_at_10_bm25"],
        "Dense alone": summary["mrr_at_10_dense"],
        "Hybrid (BM25 + Dense + RRF)": summary["mrr_at_10_hybrid"],
    }
    max_val = max(metrics.values()) if metrics else 1.0
    for label, val in metrics.items():
        bar = _ascii_bar(val, max(max_val, 0.001))
        print(f"  {label:<28} {bar}")
    print()


def print_f1_by_iterations(summary: dict):
    print("=" * 60)
    print("PLOT 2 — Mean Token F1 by Number of Agent Iterations")
    print("=" * 60)
    f1_by_iter = summary.get("mean_f1_by_iterations", {})
    if not f1_by_iter:
        print("  no iteration data found")
        print()
        return
    max_val = max(f1_by_iter.values()) if f1_by_iter else 1.0
    for n_iter in sorted(f1_by_iter.keys(), key=int):
        val = f1_by_iter[n_iter]
        bar = _ascii_bar(val, max(max_val, 0.001))
        label = f"iterations = {n_iter}"
        print(f"  {label:<28} {bar}")
    print()


def print_qualitative_examples(questions: list[dict]):
    print("=" * 60)
    print("PLOT 3 — Qualitative Examples")
    print("=" * 60)

    # find one single-hop (1 iteration) and one multi-hop (>= 3 iterations) example
    single_hop = next((q for q in questions if q["n_iterations"] == 1), None)
    multi_hop = next((q for q in questions if q["n_iterations"] >= 3), None)

    for label, example in [("single-hop (1 iteration)", single_hop), ("multi-hop (≥3 iterations)", multi_hop)]:
        if example is None:
            print(f"  [{label}] — no example found")
            print()
            continue

        print(f"\n  [{label}]")
        print(f"  Question:   {example['question'][:120]}")
        print(f"  Gold:       {example['gold_answer'][:120]}")
        print(f"  Predicted:  {example['predicted_answer'][:120]}")
        print(f"  Exact Match: {bool(example['exact_match'])}  |  Token F1: {example['token_f1']:.4f}")
        print(f"  Iterations: {example['n_iterations']}")
    print()


def print_summary_table(summary: dict):
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['n_questions']}")
    print(f"  Mean Exact Match    : {summary['mean_exact_match']:.4f}")
    print(f"  Mean Token F1       : {summary['mean_token_f1']:.4f}")
    print(f"  MRR@10 BM25         : {summary['mrr_at_10_bm25']:.4f}")
    print(f"  MRR@10 Dense        : {summary['mrr_at_10_dense']:.4f}")
    print(f"  MRR@10 Hybrid       : {summary['mrr_at_10_hybrid']:.4f}")
    print()


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"results file not found at {RESULTS_PATH}")
        print("run experiments/eval.py first")
        sys.exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    questions = data["questions"]

    print()
    print_summary_table(summary)
    print_mrr_comparison(summary)
    print_f1_by_iterations(summary)
    print_qualitative_examples(questions)


if __name__ == "__main__":
    main()

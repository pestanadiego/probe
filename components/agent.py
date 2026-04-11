from components.answer_generator import generate_answer
from components.dense import DenseRetriever
from components.memory import AgentTrace, IterationTrace, Memory
from components.orchestrator import DECISION_ANSWER, DECISION_SEARCH, decide
from components.query_generator import generate_query
from components.reformulator import reformulate_query
from components.reranker import Reranker
from components.retriever import HybridRetriever
from components.verifier import verify

MAX_ITER = 6

DEMO_CORPUS_DIR = "corpus/datasheets/text"
DEMO_INDEX_DIR = "index/demo"


def run(question: str, index_path: str = DEMO_INDEX_DIR, corpus_path: str = DEMO_CORPUS_DIR) -> AgentTrace:
    dense = DenseRetriever()
    retriever = HybridRetriever(dense)
    retriever.load_index(index_path, corpus_path)
    reranker = Reranker()
    return run_with_components(question, retriever, reranker)


def run_with_components(
    question: str,
    retriever: HybridRetriever,
    reranker: Reranker,
) -> AgentTrace:
    memory = Memory()
    iterations = []

    for i in range(MAX_ITER):
        memory.iteration_counter = i
        action = decide(question, memory)

        if action == DECISION_ANSWER:
            answer, sources = generate_answer(question, memory)
            iterations.append(IterationTrace(
                iteration=i,
                query="",
                retrieved_chunks=[],
                reranker_scores=[],
                verification="PASS",
                action=DECISION_ANSWER,
            ))
            return AgentTrace(
                question=question,
                iterations=iterations,
                final_answer=answer,
                sources=sources,
            )

        query = generate_query(question, memory)
        memory.add_query(query)

        chunks, scores, passed = _search_and_verify(query, retriever, reranker)

        if not passed:
            reformulated = reformulate_query(query, chunks)
            memory.add_query(reformulated)
            chunks, scores, passed = _search_and_verify(reformulated, retriever, reranker)

        iterations.append(IterationTrace(
            iteration=i,
            query=query,
            retrieved_chunks=chunks,
            reranker_scores=scores,
            verification="PASS" if passed else "FAIL",
            action=DECISION_SEARCH,
        ))

        if passed:
            memory.add_chunks(chunks)

    answer, sources = generate_answer(question, memory)
    return AgentTrace(
        question=question,
        iterations=iterations,
        final_answer=answer,
        sources=sources,
    )


def _search_and_verify(
    query: str,
    retriever: HybridRetriever,
    reranker: Reranker,
) -> tuple[list, list[float], bool]:
    """Run retrieval, reranking, and verification for a single query."""
    candidate_chunks = retriever.search(query, top_k=20)
    reranked_chunks, reranker_scores = reranker.rerank(query, candidate_chunks, top_k=5)
    passed = verify(reranker_scores)
    return reranked_chunks, reranker_scores, passed

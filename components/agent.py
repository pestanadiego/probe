from components.answer_generator import generate_answer
from components.memory import AgentTrace, IterationTrace, Memory
from components.orchestrator import DECISION_ANSWER, decide
from components.query_generator import generate_query
from components.reformulator import reformulate_query
from components.reranker import Reranker
from components.retriever import HybridRetriever
from components.verifier import verify

MAX_ITER = 6


def run(
    question: str,
    retriever: HybridRetriever,
    reranker: Reranker,
) -> AgentTrace:
    """Run the iterative retrieval agent and return a full trace."""
    memory = Memory()
    iteration_traces = []

    for iteration in range(MAX_ITER):
        memory.iteration_counter = iteration

        action = decide(question, memory)

        if action == DECISION_ANSWER:
            answer, sources = generate_answer(question, memory)
            # record the final answer iteration with no retrieval
            iteration_traces.append(IterationTrace(
                iteration=iteration,
                query="",
                retrieved_chunks=[],
                reranker_scores=[],
                verification="PASS",
                action=DECISION_ANSWER,
            ))
            return AgentTrace(
                question=question,
                iterations=iteration_traces,
                final_answer=answer,
                sources=sources,
            )

        # action is SEARCH
        query = generate_query(question, memory)
        memory.add_query(query)

        chunks, reranker_scores, verification_result = _search_and_verify(
            query, retriever, reranker
        )

        if not verification_result:
            # reformulate and retry once
            reformulated_query = reformulate_query(query, chunks)
            memory.add_query(reformulated_query)
            chunks, reranker_scores, verification_result = _search_and_verify(
                reformulated_query, retriever, reranker
            )

        verification_str = "PASS" if verification_result else "FAIL"

        iteration_traces.append(IterationTrace(
            iteration=iteration,
            query=query,
            retrieved_chunks=chunks,
            reranker_scores=reranker_scores,
            verification=verification_str,
            action=action,
        ))

        if verification_result:
            memory.add_chunks(chunks)

    # iteration cap reached — generate answer with whatever we have
    answer, sources = generate_answer(question, memory)
    return AgentTrace(
        question=question,
        iterations=iteration_traces,
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

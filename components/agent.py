from collections.abc import Callable

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

class Emitter:
    def __init__(self, callback: Callable[[dict], None] | None, iteration: int):
        self._callback = callback
        self._iteration = iteration

    def __call__(self, stage: str, message: str, state: str = "running") -> None:
        if self._callback:
            self._callback({
                "stage": stage,
                "message": message,
                "iteration": self._iteration,
                "state": state,
            })

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
    on_iteration=None,
    on_event: Callable[[dict], None] | None = None,
) -> AgentTrace:
    memory = Memory()
    iterations = []

    for i in range(MAX_ITER):
        emit = Emitter(on_event, i)
        memory.iteration_counter = i

        emit("decision", "Deciding whether memory is sufficient")
        action = decide(question, memory)
        emit("decision", f"Decision: {action}", state="complete")

        if action == DECISION_ANSWER:
            emit("answer", "Generating final answer from collected evidence")
            answer, sources = generate_answer(question, memory)
            emit("answer", "Final answer ready", state="complete")
            return AgentTrace(
                question=question,
                iterations=iterations,
                final_answer=answer,
                sources=sources,
            )

        emit("query", "Generating search query")
        query = generate_query(question, memory)
        emit("query", f"Query ready: {query}", state="complete")
        memory.add_query(query)

        chunks, scores, passed = _search_and_verify(query, retriever, reranker, emit)

        if not passed:
            emit("reformulate", "Initial evidence was weak, reformulating the query")
            reformulated = reformulate_query(query, chunks)
            emit("reformulate", f"Reformulated query: {reformulated}", state="complete")
            memory.add_query(reformulated)
            chunks, scores, passed = _search_and_verify(
                reformulated, retriever, reranker, emit, attempt="reformulated"
            )

        trace = IterationTrace(
            iteration=i,
            query=query,
            retrieved_chunks=chunks,
            reranker_scores=scores,
            verification="PASS" if passed else "FAIL",
            action=DECISION_SEARCH,
        )
        iterations.append(trace)
        if on_iteration:
            on_iteration(trace)

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
    emit: Emitter,
    attempt: str = "primary",
) -> tuple[list, list[float], bool]:
    emit("retrieval", f"Searching corpus with {attempt} query")
    candidate_chunks = retriever.search(query, top_k=20)
    emit("retrieval", f"Retrieved {len(candidate_chunks)} candidate chunks", state="complete")

    emit("rerank", "Reranking candidate chunks")
    reranked_chunks, reranker_scores = reranker.rerank(query, candidate_chunks, top_k=5)
    emit("rerank", f"Reranked top {len(reranked_chunks)} chunks", state="complete")

    emit("verify", "Verifying retrieval quality")
    passed = verify(reranker_scores)
    emit("verify", f"Verification: {'PASS' if passed else 'FAIL'}", state="complete" if passed else "error")

    return reranked_chunks, reranker_scores, passed

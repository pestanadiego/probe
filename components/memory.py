from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str
    score: float


@dataclass
class IterationTrace:
    iteration: int
    query: str
    retrieved_chunks: list[Chunk]
    reranker_scores: list[float]
    verification: str  # "PASS" or "FAIL"
    action: str        # "SEARCH" or "ANSWER"


@dataclass
class AgentTrace:
    question: str
    iterations: list[IterationTrace]
    final_answer: str
    sources: list[Chunk]


@dataclass
class Memory:
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    search_history: list[str] = field(default_factory=list)
    iteration_counter: int = 0
    max_chunks: int = 10

    def add_chunks(self, chunks: list[Chunk]):
        """Add new chunks, keeping only the top max_chunks by score."""
        self.retrieved_chunks.extend(chunks)
        self.retrieved_chunks.sort(key=lambda c: c.score, reverse=True)
        self.retrieved_chunks = self.retrieved_chunks[:self.max_chunks]

    def add_query(self, query: str):
        self.search_history.append(query)

    def context_text(self) -> str:
        """Return numbered context string for LLM prompts."""
        return "\n\n".join(
            f"[{i + 1}] (source: {c.source})\n{c.text}"
            for i, c in enumerate(self.retrieved_chunks)
        )

from components.llm import get_llm
from components.memory import Chunk

MAX_QUERY_WORDS = 10

def reformulate_query(original_query: str, failed_chunks: list[Chunk]) -> str:
    llm = get_llm()

    if failed_chunks:
        chunk_summary = "\n".join(
            f"- {c.text[:150]}..." if len(c.text) > 150 else f"- {c.text}"
            for c in failed_chunks[:3]
        )
    else:
        chunk_summary = "no relevant chunks were returned"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query reformulator for a technical document retrieval system.\n\n"
                "Rules:\n"
                f"- Output only a new search query, 1–{MAX_QUERY_WORDS} words\n"
                "- No punctuation, no quotes, no explanation\n"
                "- The new query must approach the topic from a completely different angle\n"
                "- Do not use synonyms of the original query — find a new perspective\n"
                "- Focus on related technical concepts, register names, or alternate terminology"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query that failed: {original_query}\n\n"
                f"Low-quality chunks that were returned:\n{chunk_summary}\n\n"
                "Generate a fundamentally different search query:"
            ),
        },
    ]

    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = llm.generate(prompt, max_new_tokens=30)

    query = response.strip().strip("\"'.,;:!?")
    words = query.split()

    if not words:
        return " ".join(reversed(original_query.split()[:MAX_QUERY_WORDS]))

    if len(words) > MAX_QUERY_WORDS:
        words = words[:MAX_QUERY_WORDS]

    return " ".join(words)

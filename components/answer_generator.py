from components.llm import get_llm
from components.memory import Chunk, Memory

def generate_answer(question: str, memory: Memory) -> tuple[str, list[Chunk]]:
    llm = get_llm()

    numbered_context = memory.context_text()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant answering questions about embedded systems.\n\n"
                "Rules:\n"
                "- Answer using ONLY the information in the provided context chunks\n"
                "- Cite chunk numbers inline using [1], [2], etc. format\n"
                "- Be specific: include register names, values, and procedures when available\n"
                "- If the context does not contain enough information, say so explicitly\n"
                "- Do not add information not present in the context"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Context:\n{numbered_context}\n\n"
                "Answer (cite chunk numbers inline with [N]):"
            ),
        },
    ]

    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    answer = llm.generate(prompt, max_new_tokens=512)

    # identify which chunks were cited in the answer
    cited_chunks = _extract_cited_chunks(answer, memory.retrieved_chunks)

    return answer, cited_chunks

def _extract_cited_chunks(answer: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return chunks whose citation numbers appear in the answer text."""
    cited = []
    for i, chunk in enumerate(chunks):
        citation = f"[{i + 1}]"
        if citation in answer:
            cited.append(chunk)

    # if no citations found, return all chunks as sources
    if not cited:
        return list(chunks)

    return cited

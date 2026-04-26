from components.llm import get_llm
from components.memory import Memory

DECISION_SEARCH = "SEARCH"
DECISION_ANSWER = "ANSWER"

def decide(question: str, memory: Memory) -> str:
    llm = get_llm()

    context_summary = (
        f"chunks collected: {len(memory.retrieved_chunks)}/{memory.max_chunks}\n"
        f"searches so far: {memory.iteration_counter}\n"
        f"queries used: {', '.join(memory.search_history) if memory.search_history else 'none'}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval controller deciding whether to search for more information "
                "or answer the question using the current context.\n\n"
                "Rules:\n"
                "- Output exactly one word: SEARCH or ANSWER\n"
                "- Output ANSWER only if the current context is sufficient to fully answer the question\n"
                "- Output SEARCH if more information is needed\n"
                "- No explanation, no punctuation — just the single word"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Current retrieval state:\n{context_summary}\n\n"
                f"Current context:\n{memory.context_text() if memory.retrieved_chunks else 'No context yet.'}\n\n"
                "Decision (SEARCH or ANSWER):"
            ),
        },
    ]

    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = llm.generate(prompt, max_new_tokens=10)

    first_word = response.strip().split()[0].upper() if response.strip() else ""

    if first_word == DECISION_ANSWER:
        return DECISION_ANSWER

    return DECISION_SEARCH # if parsing fails

from components.llm import get_llm
from components.memory import Memory

MAX_QUERY_WORDS = 10
MIN_QUERY_WORDS = 1

def generate_query(question: str, memory: Memory) -> str:
    llm = get_llm()

    history_text = (
        "\n".join(f"- {q}" for q in memory.search_history)
        if memory.search_history
        else "none"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query generator for a technical document retrieval system.\n\n"
                "Rules:\n"
                f"- Output only the search query, {MIN_QUERY_WORDS}–{MAX_QUERY_WORDS} words\n"
                "- No punctuation, no quotes, no explanation\n"
                "- Do not repeat any previous query\n"
                "- Focus on a specific technical aspect not yet covered"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Previous queries (do not repeat these):\n{history_text}\n\n"
                "Generate a new search query:"
            ),
        },
    ]

    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = llm.generate(prompt, max_new_tokens=30)

    query = response.strip().strip("\"'.,;:!?")
    words = query.split()

    if len(words) < MIN_QUERY_WORDS:
        return " ".join(question.split()[:MAX_QUERY_WORDS])

    if len(words) > MAX_QUERY_WORDS:
        words = words[:MAX_QUERY_WORDS]

    return " ".join(words)

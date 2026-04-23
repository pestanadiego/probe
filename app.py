import streamlit as st

from components.agent import DEMO_CORPUS_DIR, DEMO_INDEX_DIR, run_with_components
from components.dense import DenseRetriever
from components.llm import get_llm
from components.reranker import Reranker
from components.retriever import HybridRetriever

from dotenv import load_dotenv

load_dotenv()

SINGLE_HOP_QUESTIONS = [
    "What is the maximum clock frequency of the ESP32?",
    "How many GPIO pins does the STM32F4 have?",
]

MULTI_HOP_QUESTIONS = [
    "Given an 80MHz APB clock on the ESP32, what register value do I write to get 400kHz on I2C fast mode?",
    "What is the maximum SPI clock speed on the ESP32, and does it change depending on which pins you use?",
    "On the STM32, if I want to use DMA with UART, which DMA channels are valid and what is the setup sequence?",
]


@st.cache_resource
def load_llm():
    return get_llm()


@st.cache_resource
def load_retriever():
    dense = DenseRetriever()
    retriever = HybridRetriever(dense)
    retriever.load_index(DEMO_INDEX_DIR, DEMO_CORPUS_DIR)
    return retriever


@st.cache_resource
def load_reranker():
    return Reranker()


def render_iteration(container, trace):
    verdict = trace.verification
    icon = "pass" if verdict == "PASS" else "error"
    label = f"Iteration {trace.iteration + 1} — {trace.action}"
    if trace.query:
        label += f": {trace.query}"

    with container.status(label, state="complete"):
        if trace.action == "ANSWER":
            st.write("Generating final answer from collected evidence.")
            return

        st.markdown(f"**Query:** {trace.query}")

        if trace.reranker_scores:
            scores_str = ", ".join(f"{s:.4f}" for s in trace.reranker_scores)
            st.markdown(f"**Reranker scores:** [{scores_str}]")

        if verdict == "PASS":
            st.success(f"Verification: {verdict}")
        else:
            st.error(f"Verification: {verdict}")

        if trace.retrieved_chunks:
            with st.expander(f"Top chunks ({len(trace.retrieved_chunks)})"):
                for i, chunk in enumerate(trace.retrieved_chunks):
                    score_display = f"{chunk.score:.4f}" if chunk.score else "—"
                    st.markdown(f"**[{i + 1}]** *{chunk.source}* (score: {score_display})")
                    st.text(chunk.text[:300])
                    if i < len(trace.retrieved_chunks) - 1:
                        st.divider()


def render_answer(trace):
    st.markdown("---")
    st.subheader("Answer")
    st.markdown(trace.final_answer)

    if trace.sources:
        st.markdown("---")
        st.subheader("Sources")
        for i, source in enumerate(trace.sources):
            with st.expander(f"[{i + 1}] {source.source} (score: {source.score:.4f})"):
                st.text(source.text[:500])


def main():
    st.set_page_config(page_title="PROBE", page_icon="~", layout="wide")
    st.title("PROBE")
    st.caption("Iterative retrieval agent for multi-hop technical question answering")

    # load models once
    load_llm()
    retriever = load_retriever()
    reranker = load_reranker()

    # question input
    question = st.text_input("Ask a question about embedded systems datasheets:")

    # preset question buttons
    st.markdown("**Single-hop**")
    cols = st.columns(len(SINGLE_HOP_QUESTIONS))
    for col, q in zip(cols, SINGLE_HOP_QUESTIONS):
        if col.button(q, key=f"single_{q[:20]}"):
            st.session_state["question"] = q

    st.markdown("**Multi-hop**")
    cols = st.columns(len(MULTI_HOP_QUESTIONS))
    for col, q in zip(cols, MULTI_HOP_QUESTIONS):
        if col.button(q, key=f"multi_{q[:20]}"):
            st.session_state["question"] = q

    # resolve which question to run
    active_question = question or st.session_state.get("question", "")

    if not active_question:
        return

    st.markdown("---")
    st.subheader("Agent Trace")

    # container for live iteration rendering
    trace_container = st.container()

    def on_iteration(iteration_trace):
        render_iteration(trace_container, iteration_trace)

    with st.spinner("Running PROBE agent..."):
        agent_trace = run_with_components(
            active_question, retriever, reranker, on_iteration=on_iteration,
        )

    render_answer(agent_trace)


if __name__ == "__main__":
    main()

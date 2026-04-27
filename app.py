import html
import os

import streamlit as st
from dotenv import load_dotenv

from components.agent import DEMO_CORPUS_DIR, DEMO_INDEX_DIR, run_with_components
from components.dense import DenseRetriever
from components.llm import get_llm
from components.reranker import Reranker
from components.retriever import HybridRetriever

load_dotenv()

STAGE_LABELS = {
    "decision": "Deciding",
    "query": "Generating query",
    "retrieval": "Retrieving",
    "rerank": "Reranking",
    "verify": "Verifying",
    "reformulate": "Reformulating",
    "answer": "Generating answer",
}

EXAMPLE_QUESTIONS = [
    "What is the maximum clock frequency of the ESP32?",
    "How many GPIO pins does the STM32F4 have?",
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


def set_question_input(question: str):
    st.session_state["question_input"] = question


def clear_question_input():
    st.session_state["question_input"] = ""


def inject_styles():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    with open(css_path) as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " ").title())


class RunCallbacks:
    def __init__(self, status_container, trace_container, active_question):
        self.events = []
        self._status = status_container
        self._trace = trace_container
        self._question = active_question

    def on_event(self, event):
        self.events.append(event)
        render_status(self._status, self.events)

    def on_iteration(self, trace):
        render_iteration(self._trace, trace)


def render_status(container, events):
    latest = events[-1] if events else None
    stage = stage_label(latest["stage"]) if latest else "Initializing"
    state = latest["state"] if latest else "running"
    iteration = latest["iteration"] + 1 if latest and latest.get("iteration") is not None else None
    completed = len([e for e in events if e.get("stage") == "decision" and e.get("state") == "complete"])
    progress_pct = min(100, int(completed / 6 * 100)) if completed else 3

    state_class = state if state in {"running", "complete", "error"} else "running"
    iter_text = f" — iteration {iteration} of 6" if iteration else ""

    container.empty()
    with container.container():
        st.markdown(
            f"<div class='probe-progress-track'><div class='probe-progress-fill'></div></div>"
            f"<div class='probe-status-line {html.escape(state_class)}'>"
            f"<span class='probe-status-dot'></span>"
            f"<span class='probe-status-stage'>{html.escape(stage)}</span>"
            f"<span class='probe-status-iter'>{html.escape(iter_text)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_iteration(container, trace):
    if trace.action == "ANSWER":
        return

    label = f"Iteration {trace.iteration + 1} ({trace.action})"
    if trace.query:
        label += f": {trace.query}"

    with container.expander(label, expanded=False):
        st.markdown(f"**Query:** {trace.query}")

        if trace.reranker_scores:
            scores_str = ", ".join(f"{s:.4f}" for s in trace.reranker_scores)
            st.markdown(f"**Reranker scores:** [{scores_str}]")

        verdict = trace.verification
        verdict_class = "pass" if verdict == "PASS" else "fail"
        verdict_mark = "✓" if verdict == "PASS" else "✕"
        st.markdown(
            f"<div class='probe-verdict {verdict_class}'>"
            f"<span class='probe-verdict-mark'>{verdict_mark}</span>"
            f"<span>Verification — {html.escape(verdict)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if trace.retrieved_chunks:
            with st.expander(f"Top chunks ({len(trace.retrieved_chunks)})"):
                for i, chunk in enumerate(trace.retrieved_chunks):
                    score_display = f"{chunk.score:.4f}" if chunk.score else "—"
                    st.markdown(f"**[{i + 1}]** *{chunk.source}* (score: {score_display})")
                    st.text(chunk.text[:300])
                    if i < len(trace.retrieved_chunks) - 1:
                        st.divider()


def render_answer(container, trace):
    container.empty()
    with container.container():
        st.markdown(
            f"<div class='probe-answer'>"
            f"<div class='probe-answer-body'>{trace.final_answer}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if trace.sources:
            footnotes_html = "<div class='probe-footnotes'>"
            footnotes_html += "<div class='probe-footnotes-label'>Sources</div>"
            for i, source in enumerate(trace.sources):
                snippet = html.escape(source.text[:280]).replace("\n", " ")
                footnotes_html += (
                    f"<div class='probe-footnote'>"
                    f"<div class='probe-footnote-mark'>[{i + 1}]</div>"
                    f"<div>"
                    f"<span class='probe-footnote-source'>{html.escape(source.source)}</span>"
                    f"<span class='probe-footnote-score'>· score {source.score:.4f}</span>"
                    f"<div class='probe-footnote-text'>{snippet}…</div>"
                    f"</div>"
                    f"</div>"
                )
            footnotes_html += "</div>"
            st.markdown(footnotes_html, unsafe_allow_html=True)


def render_pre_query():
    st.markdown(
        "<div class='probe-prequery'>"
        "<h1 class='probe-masthead'>Probe</h1>"
        "<p class='probe-tagline'>Iterative retrieval agent for multi-hop technical question answering</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 5, 1])
    with cols[1]:
        st.text_input(
            "search",
            key="question_input",
            label_visibility="collapsed",
            placeholder="Ask a question…",
        )

        with st.expander("Example questions", expanded=False):
            for q in EXAMPLE_QUESTIONS:
                st.button(q, key=f"ex_{q[:24]}", on_click=set_question_input, args=(q,))

def render_post_query(active_question, retriever, reranker):
    st.button("← New search", key="new_search", on_click=clear_question_input)

    st.markdown(
        f"<h1 class='probe-headline'>{html.escape(active_question)}</h1>",
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()
    answer_placeholder = st.empty()

    with answer_placeholder.container():
        st.markdown(
            "<div class='probe-loading'>"
            "<div class='probe-spinner'></div>"
            "<span>Reasoning over the corpus…</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='probe-trace-section'></div>", unsafe_allow_html=True)
    with st.expander("Show trace", expanded=False):
        trace_container = st.container()

    callbacks = RunCallbacks(status_placeholder, trace_container, active_question)
    render_status(status_placeholder, callbacks.events)

    agent_trace = run_with_components(
        active_question,
        retriever,
        reranker,
        on_iteration=callbacks.on_iteration,
        on_event=callbacks.on_event,
    )

    render_status(status_placeholder, callbacks.events)
    render_answer(answer_placeholder, agent_trace)


def main():
    st.set_page_config(
        page_title="Probe",
        page_icon="🔎",
        layout="centered",
    )
    inject_styles()

    load_llm()
    retriever = load_retriever()
    reranker = load_reranker()

    active_question = st.session_state.get("question_input", "").strip()

    if not active_question:
        render_pre_query()
        return

    render_post_query(active_question, retriever, reranker)


if __name__ == "__main__":
    main()

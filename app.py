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


def set_question_input(question: str):
    st.session_state["question_input"] = question


def inject_styles():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    with open(css_path) as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " ").title())


class RunCallbacks:
    def __init__(self, activity_container, trace_container, active_question):
        self.events = []
        self._activity = activity_container
        self._trace = trace_container
        self._question = active_question

    def on_event(self, event):
        self.events.append(event)
        render_live_activity(self._activity, self.events, self._question)

    def on_iteration(self, trace):
        render_iteration(self._trace, trace)


def render_live_activity(container, events, active_question):
    latest_event = events[-1] if events else None
    current_message = latest_event["message"] if latest_event else "Waiting for the agent to start"
    current_stage = stage_label(latest_event["stage"]) if latest_event else "Idle"
    current_state = latest_event["state"] if latest_event else "running"
    current_iteration = latest_event["iteration"] + 1 if latest_event and latest_event["iteration"] is not None else None
    total_iterations = len([e for e in events if e.get("stage") == "decision" and e.get("state") == "complete"])
    progress_value = min(1.0, total_iterations / 6) if total_iterations else 0.02
    running_state = current_state if current_state in {"running", "complete", "error"} else "running"
    status_icon = "◉" if running_state == "running" else ("✓" if running_state == "complete" else "!")

    progress_pct = min(100, int(total_iterations / 6 * 100)) if total_iterations else 2
    iteration_meta = f" · iteration {current_iteration}/6" if current_iteration else ""

    container.empty()

    with container.container():
        st.markdown(
            "<div class='probe-panel'>"
            "<h3>Live status</h3>"
            f"<div class='probe-progress-track'><div class='probe-progress-fill' style='width:{progress_pct}%'></div></div>"
            "<div class='probe-status-row'>"
            f"<div class='probe-status-icon {html.escape(running_state)}'>{html.escape(status_icon)}</div>"
            "<div class='probe-status-copy'>"
            f"<div class='probe-status-label'>Now running · {html.escape(current_stage)}</div>"
            f"<div class='probe-status-message'>{html.escape(current_message)}</div>"
            f"<div class='probe-status-meta'>{html.escape(current_state.upper())}{html.escape(iteration_meta)}</div>"
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Steps", expanded=False):
            st.caption(active_question)

            step_events = []
            step_index = {}
            tracked_stages = {"decision", "query", "retrieval", "rerank", "verify", "reformulate", "answer"}
            for event in events:
                stage = event["stage"]
                iteration = event.get("iteration")
                key = (iteration, stage)
                if stage in tracked_stages:
                    if key not in step_index:
                        step_index[key] = len(step_events)
                        step_events.append(event)
                    else:
                        step_events[step_index[key]] = event

            if step_events:
                st.markdown("<div class='probe-step-list'>", unsafe_allow_html=True)
                for event in step_events[-12:]:
                    state_class = event["state"]
                    iteration_text = (
                        f"Iteration {event['iteration'] + 1}" if event.get("iteration") is not None else ""
                    )
                    bullet = "✓" if state_class == "complete" else ("◌" if state_class == "running" else "!")
                    st.markdown(
                        f"<div class='probe-step-item {html.escape(state_class)}'>"
                        f"<div class='probe-step-bullet {html.escape(state_class)}'>{html.escape(bullet)}</div>"
                        "<div style='min-width:0; flex:1 1 auto;'>"
                        f"<div class='probe-step-text'>{html.escape(event['message'])}</div>"
                        f"<div class='probe-step-meta'>{html.escape(stage_label(event['stage']))}"
                        f"{f' · {html.escape(iteration_text)}' if iteration_text else ''}</div>"
                        "</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("Steps will appear here as the agent runs.")


def render_iteration(container, trace):
    if trace.action == "ANSWER":
        return

    label = f"Iteration {trace.iteration + 1} — {trace.action}"
    if trace.query:
        label += f": {trace.query}"

    with container.expander(label, expanded=False):
        st.markdown(f"**Query:** {trace.query}")

        if trace.reranker_scores:
            scores_str = ", ".join(f"{s:.4f}" for s in trace.reranker_scores)
            st.markdown(f"**Reranker scores:** [{scores_str}]")

        verdict = trace.verification
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


def render_answer(container, trace):
    container.empty()
    with container.container():
        st.subheader("Answer")
        st.markdown(trace.final_answer)

        if trace.sources:
            with st.expander(f"Sources ({len(trace.sources)})", expanded=False):
                for i, source in enumerate(trace.sources):
                    st.markdown(f"**[{i + 1}] {source.source}** — score: {source.score:.4f}")
                    st.text(source.text[:500])
                    if i < len(trace.sources) - 1:
                        st.divider()


def main():
    st.set_page_config(page_title="PROBE", page_icon="~", layout="wide")
    inject_styles()

    st.markdown(
        "<div class='probe-hero'>"
        "<h1>PROBE</h1>"
        "<p>Iterative retrieval agent for multi-hop technical question answering</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    load_llm()
    retriever = load_retriever()
    reranker = load_reranker()

    question = st.text_input(
        "Ask a question about embedded systems datasheets:",
        key="question_input",
    )

    st.markdown("**Single-hop**")
    cols = st.columns(len(SINGLE_HOP_QUESTIONS))
    for col, q in zip(cols, SINGLE_HOP_QUESTIONS):
        col.button(q, key=f"single_{q[:20]}", on_click=set_question_input, args=(q,))

    st.markdown("**Multi-hop**")
    cols = st.columns(len(MULTI_HOP_QUESTIONS))
    for col, q in zip(cols, MULTI_HOP_QUESTIONS):
        col.button(q, key=f"multi_{q[:20]}", on_click=set_question_input, args=(q,))

    active_question = question
    if not active_question:
        return

    st.markdown("---")

    answer_placeholder = st.empty()
    with answer_placeholder.container():
        st.subheader("Answer")
        st.markdown(
            "<div class='probe-loading'>"
            "<div class='probe-spinner'></div>"
            "<span>Agent is reasoning over the corpus...</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    activity_container = st.empty()

    with st.expander("Iteration Trace", expanded=False):
        trace_container = st.container()

    callbacks = RunCallbacks(activity_container, trace_container, active_question)
    render_live_activity(activity_container, callbacks.events, active_question)

    with st.spinner("Running PROBE agent..."):
        agent_trace = run_with_components(
            active_question,
            retriever,
            reranker,
            on_iteration=callbacks.on_iteration,
            on_event=callbacks.on_event,
        )

    render_live_activity(activity_container, callbacks.events, active_question)
    render_answer(answer_placeholder, agent_trace)


if __name__ == "__main__":
    main()

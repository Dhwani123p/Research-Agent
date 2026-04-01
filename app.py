"""Streamlit frontend for the Personal Research Agent."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.status-box {
    background: #1e1e2e;
    border-left: 3px solid #7c3aed;
    padding: 8px 14px;
    border-radius: 4px;
    margin: 3px 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #c4b5fd;
}
.note-box {
    background: #1e2a1e;
    border-left: 3px solid #16a34a;
    padding: 8px 14px;
    border-radius: 4px;
    margin: 3px 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #86efac;
}
.report-container {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 24px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Personal Research Agent")
st.caption("Powered by Gemini 2.0 Flash · Tavily · ChromaDB · ArXiv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Free key at aistudio.google.com/apikey",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Free key at tavily.com",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. **Planner** breaks your query into sub-tasks
2. **Executor** uses tools (web, ArXiv, PDFs) to gather facts
3. **Critic** checks if research is complete — loops back if not
4. **Synthesizer** writes a cited report
5. Results saved to **long-term memory** for future sessions
    """)

    st.divider()
    if st.button("Clear Memory", use_container_width=True):
        import shutil
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        st.success("Long-term memory cleared.")

# ── Main input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "Research Query",
    placeholder='e.g. "What is the competitive landscape of solid-state EV batteries in 2025?"',
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("Research", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a research query.")
        st.stop()
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Gemini API key is required.")
        st.stop()
    if not os.getenv("TAVILY_API_KEY"):
        st.error("Tavily API key is required.")
        st.stop()

    from agent.orchestrator import ResearchOrchestrator

    st.divider()

    log_col, report_col = st.columns([1, 2])

    with log_col:
        st.subheader("Agent Log")
        log_container = st.container()

    with report_col:
        st.subheader("Research Report")
        report_placeholder = st.empty()

    log_entries = []

    with st.spinner("Agent is working..."):
        orchestrator = ResearchOrchestrator()
        for event in orchestrator.run(query):
            if event["type"] == "status":
                log_entries.append(f'<div class="status-box">⚙ {event["content"]}</div>')
            elif event["type"] == "note":
                log_entries.append(f'<div class="note-box">📎 {event["content"]}</div>')
            elif event["type"] == "report":
                report_placeholder.markdown(
                    f'<div class="report-container">{event["content"]}</div>',
                    unsafe_allow_html=True,
                )
            elif event["type"] == "error":
                log_entries.append(f'<div class="status-box" style="border-color:#ef4444;color:#fca5a5;">✗ {event["content"]}</div>')

            with log_container:
                st.markdown("\n".join(log_entries), unsafe_allow_html=True)

    st.success("Research complete. Report saved to long-term memory.")

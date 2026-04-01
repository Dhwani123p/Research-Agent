---
title: Research Agent
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Personal Research Agent

An agentic AI research assistant that plans, searches, evaluates, and synthesises information — so you can ask a complex question and walk away.

## Architecture

```
Query → Planner → sub-tasks
              ↓
    ┌─────────────────────────┐
    │  Memory Store           │ ←→ Tool Executor (web, ArXiv, PDFs)
    │  (ChromaDB + context)   │ ←→ Synthesiser (cross-reference)
    └─────────────────────────┘
              ↓
    Critic / Evaluator ──(incomplete)──→ loop back
              ↓ (complete)
    Report Generator → cited report
```

**Layer 1 — Tools**
- `web_search` — Tavily API for real-time web search
- `fetch_page` — full webpage content extraction
- `arxiv_search` — academic paper search
- `read_pdf` — PDF text extraction (local or URL)

**Layer 2 — Memory**
- Short-term: session notes held in context
- Long-term: ChromaDB vector DB with `all-MiniLM-L6-v2` embeddings

**Layer 3 — Orchestration (ReAct)**
- Planner → Executor → Critic → Synthesizer loop
- Claude Sonnet 4.6 with native tool use

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/Dhwani123p/Research-Agent.git
cd Research-Agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

You need:
- **Anthropic API key** — [console.anthropic.com](https://console.anthropic.com)
- **Tavily API key** — [tavily.com](https://tavily.com) (free tier available)

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example Queries

- *"Research the competitive landscape of solid-state EV batteries in 2025"*
- *"What are the latest breakthroughs in GLP-1 receptor agonists for obesity?"*
- *"Compare transformer and Mamba architectures for long-context language modeling"*

## Project Structure

```
Research-Agent/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── agent/
│   ├── orchestrator.py     # Main ReAct loop
│   └── prompts.py          # System prompts
├── tools/
│   ├── web_search.py       # Tavily
│   ├── fetch_page.py       # BeautifulSoup scraper
│   ├── arxiv_search.py     # ArXiv API
│   └── pdf_reader.py       # PyPDF2
└── memory/
    ├── short_term.py       # Session context
    └── long_term.py        # ChromaDB persistence
```

# 🔍 Multi-Domain Research Agent

An **agentic workflow** that classifies any research query, routes it to the right pipeline, autonomously calls multiple tools, self-reflects on whether it has enough information, and returns a **typed Pydantic report** — built with **LangGraph**, **Pydantic**, and **Claude**.

> Submission for the Lumiq take-home (Option A — Research Agent).

---

## ✨ What it does

The agent handles **three distinct research shapes**, each with its own structured output:

| Query shape | Example | Returns |
|---|---|---|
| 🏢 **Company** | "Razorpay" | `CompanyBrief` — industry, products, competitors, funding, SWOT, news |
| 📚 **Topic** | "Carbon capture technology" | `TopicReport` — concepts, consensus, debates, findings, recent developments |
| ⚖️ **Comparison** | "React vs Vue" | `ComparisonReport` — side-by-side dimensions, use cases, verdict |
| ⛔ **Unsupported** | "Should I quit my job?" | `RefusalReport` — polite redirect, no fabricated research |

A **router node** classifies every incoming query and dispatches it to the appropriate pipeline. Each pipeline shares the same executor / reflector / tool layer but uses **type-specific prompts and schemas** for planning and synthesis.

This isn't a single LLM call dressed up as an "agent" — it's a proper LangGraph state machine with a real **reflection loop** and a real **routing decision**.

---

## 🏗️ Architecture

<img width="680" height="656" alt="research_agent_architecture" src="https://github.com/user-attachments/assets/77bfdac8-5918-4e7c-9890-1091a931a4bd" />

---

### Why this design

| Decision | Rationale |
|---|---|
| **Router node first** | Lets one agent serve multiple research shapes without prompt entanglement. Refusal becomes an explicit path, not a fragile guardrail buried in a prompt. |
| **Type-specific prompts + schemas** | A company brief, a topic report, and a comparison are genuinely different deliverables. Forcing one schema for all would water down each. |
| **Shared executor + tools** | The *act* of searching the web is the same regardless of query type. Sharing this layer keeps the codebase small. |
| **LangGraph state machine** (not a ReAct loop) | Each step is explicit and inspectable — reflection is a real conditional edge, not buried in a system prompt. Easier to demo. |
| **Pydantic-typed output** | Schema enforcement means UI never has to defensively check for missing keys, and any synthesis bug surfaces immediately as a validation error. |
| **Capped iterations (3)** | Bounds latency and cost without removing the agentic loop. |
| **Tool failures don't crash** | Each tool wrapper returns `{error: ...}` instead of raising, so the agent can still synthesize from partial data. |
| **Streamlit, not FastAPI** | The report is human-consumed. Streamlit is far faster to build a credible demo UI in. |

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/someshsce/research-agent.git
cd research-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Then edit .env:
#   ANTHROPIC_API_KEY  — https://console.anthropic.com
#   TAVILY_API_KEY     — https://tavily.com (free tier is enough)
```

### 3. Run

**Streamlit UI (recommended for the demo):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python run_cli.py "Razorpay"
python run_cli.py "React vs Vue"
python run_cli.py "Carbon capture technology"
python run_cli.py "Should I quit my job?"   # demonstrates the refusal path
```

---

## 🗂️ Project structure

```
research-agent/
├── app.py                 # Streamlit UI with renderers per report type
├── run_cli.py             # CLI runner
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── LICENSE
└── agent/
    ├── __init__.py
    ├── graph.py           # LangGraph wiring + 2 conditional edges
    ├── nodes.py           # router · planner · executor · reflector · synthesizer · refusal
    ├── tools.py           # Tavily + Wikipedia wrappers
    ├── prompts.py         # router prompt + 9 type-specific prompts
    └── schemas.py         # Pydantic models for all 4 report types + AgentState
```

---

## 🔧 End-to-end trace of a single run

**Input:** `"React vs Vue"`

1. **Router** — Claude classifies as `comparison`, extracts `item_a="React"`, `item_b="Vue"`.
2. **Planner** — selects the *comparison* prompt; generates a plan such as:
   ```json
   {
     "tasks": [
       {"tool": "wikipedia",  "query": "React JavaScript library",   "rationale": "Background on React"},
       {"tool": "wikipedia",  "query": "Vue.js",                     "rationale": "Background on Vue"},
       {"tool": "web_search", "query": "React vs Vue comparison 2025", "rationale": "Direct comparison"},
       {"tool": "web_search", "query": "Vue.js use cases when to choose", "rationale": "Use cases"}
     ]
   }
   ```
3. **Executor** — runs each tool, collects results in state.
4. **Reflector** — uses the *comparison* completeness criteria. Either approves synthesis or queues 1–2 follow-up searches. Hard cap at 3 iterations.
5. **Synthesizer** — uses the *comparison* schema (`ComparisonReport`). Output is `model_validate`d, so any drift in JSON shape fails loudly.
6. **UI** — calls `render_comparison()` based on `isinstance(report, ComparisonReport)`.

---

## ⚠️ Known limitation (and what I'd do next)

**Limitation:** the agent's view of the world is whatever Tavily + Wikipedia return. For private companies, niche topics, or non-English-market subjects, public web coverage is sparse and the report reflects that — funding numbers may be stale, comparison dimensions may miss recent feature additions, debates may miss recent papers. The `confidence_score` is *self-reported* by the synthesizer, not externally validated.

**What I'd build next, in order:**
1. **Paid data integrations** — Crunchbase / Tracxn for company funding ground-truth, SEC EDGAR for public companies, Semantic Scholar for topic citations.
2. **Cross-source verification** — when two tools disagree on a fact, surface the conflict explicitly instead of silently picking one. Confidence becomes evidence-backed.
3. **Memory / caching** — persist reports in a vector store so common queries return in seconds and trends can be tracked over time.
4. **A fourth report type — `PersonProfile`** — currently folded into topic; merits its own schema (career, notable works, affiliations, quotes).
5. **Eval harness** — a small golden set per report type to track regression on prompt/model changes.

---

## 🧪 Bonus criteria covered

- ✅ **LangGraph** — full state machine with two conditional edges (router → planner/refusal, reflector → executor/synthesizer)
- ✅ **Pydantic** — typed agent state, typed routing decision, and one typed schema per report type
- ✅ **Claude** — Sonnet 4.5 across all reasoning steps (routing, planning, reflection, synthesis)

---

## 📝 License

MIT

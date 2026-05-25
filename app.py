"""Streamlit UI for the multi-domain Research Agent.

Renders 4 different output types based on what the router classified the query as:
  - CompanyBrief
  - TopicReport
  - ComparisonReport
  - RefusalReport (for unsupported queries)
"""
from __future__ import annotations

import json

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.schemas import (  # noqa: E402
    CompanyBrief,
    ComparisonReport,
    RefusalReport,
    TopicReport,
)

st.set_page_config(
    page_title="Multi-Domain Research Agent",
    page_icon="🔍",
    layout="wide",
)

# ───────────────────────────── Sidebar ─────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Research Agent")
    st.markdown(
        "An **agentic workflow** that routes any research query to the "
        "right pipeline and returns a *typed*, structured report."
    )
    st.markdown("---")
    st.markdown("### Supported research types")
    st.markdown(
        """
- 🏢 **Companies** — competitive intel briefs
- 📚 **Topics** — concepts, issues, phenomena
- ⚖️ **Comparisons** — X vs Y
- ⛔ **Other** — politely refused
"""
    )
    st.markdown("### How it works")
    st.markdown(
        """
1. **Router** — classifies the query
2. **Planner** — picks tools to call (type-aware)
3. **Executor** — runs Web / News / Wikipedia searches
4. **Reflector** — self-evaluates: enough info?
5. **Synthesizer** — builds a typed Pydantic report
"""
    )
    st.markdown("### Stack")
    st.markdown("LangGraph · Pydantic · Claude · Tavily · Wikipedia")

    st.markdown("---")
    st.markdown("### Try a sample")
    st.markdown("**Companies**")
    for s in ["Razorpay", "Anthropic", "Zerodha"]:
        if st.button(s, use_container_width=True, key=f"co_{s}"):
            st.session_state["query"] = s
    st.markdown("**Topics**")
    for s in ["Carbon capture technology", "Effects of intermittent fasting"]:
        if st.button(s, use_container_width=True, key=f"to_{s}"):
            st.session_state["query"] = s
    st.markdown("**Comparisons**")
    for s in ["React vs Vue", "Stripe vs Razorpay"]:
        if st.button(s, use_container_width=True, key=f"cm_{s}"):
            st.session_state["query"] = s
    st.markdown("**Unsupported (try it!)**")
    for s in ["Should I quit my job?", "What's the weather today?"]:
        if st.button(s, use_container_width=True, key=f"un_{s}"):
            st.session_state["query"] = s

    missing = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        st.error(f"Missing env vars: {', '.join(missing)}")


# ───────────────────────────── Renderers ─────────────────────────────


def render_company(brief: CompanyBrief) -> None:
    st.header(f"🏢 {brief.company_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Industry", brief.industry or "—")
    c2.metric("Founded", str(brief.founded) if brief.founded else "—")
    c3.metric("Confidence", f"{brief.confidence_score:.0%}")

    if brief.headquarters:
        st.markdown(f"**Headquarters:** {brief.headquarters}")
    st.markdown(f"**Description:** {brief.description}")

    if brief.key_products:
        st.subheader("🛍️ Key Products")
        for p in brief.key_products:
            st.markdown(f"- {p}")

    if brief.competitors:
        st.subheader("🥊 Competitors")
        st.markdown(", ".join(brief.competitors))

    f = brief.funding
    if any([f.total_raised, f.latest_round, f.valuation, f.notable_investors]):
        st.subheader("💰 Funding")
        if f.total_raised:
            st.markdown(f"- **Total raised:** {f.total_raised}")
        if f.latest_round:
            st.markdown(f"- **Latest round:** {f.latest_round}")
        if f.valuation:
            st.markdown(f"- **Valuation:** {f.valuation}")
        if f.notable_investors:
            st.markdown(f"- **Investors:** {', '.join(f.notable_investors)}")

    if brief.recent_news:
        st.subheader("📰 Recent News")
        for n in brief.recent_news:
            label = n.headline + (f"  ({n.date})" if n.date else "")
            with st.expander(label):
                st.markdown(n.summary)
                if n.source_url:
                    st.markdown(f"[Open source]({n.source_url})")

    swot = brief.swot
    if any([swot.strengths, swot.weaknesses, swot.opportunities, swot.threats]):
        st.subheader("🎯 SWOT")
        cL, cR = st.columns(2)
        with cL:
            if swot.strengths:
                st.markdown("**Strengths**")
                for x in swot.strengths:
                    st.markdown(f"- {x}")
            if swot.opportunities:
                st.markdown("**Opportunities**")
                for x in swot.opportunities:
                    st.markdown(f"- {x}")
        with cR:
            if swot.weaknesses:
                st.markdown("**Weaknesses**")
                for x in swot.weaknesses:
                    st.markdown(f"- {x}")
            if swot.threats:
                st.markdown("**Threats**")
                for x in swot.threats:
                    st.markdown(f"- {x}")


def render_topic(report: TopicReport) -> None:
    st.header(f"📚 {report.topic}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Domain", report.domain or "—")
    c2.metric("Iterations", report.research_iterations)
    c3.metric("Confidence", f"{report.confidence_score:.0%}")

    if report.executive_summary:
        st.subheader("📋 Executive Summary")
        st.markdown(report.executive_summary)

    if report.key_concepts:
        st.subheader("🔑 Key Concepts")
        for kc in report.key_concepts:
            with st.expander(kc.name):
                st.markdown(kc.explanation)

    if report.current_consensus:
        st.subheader("🤝 Current Consensus")
        st.markdown(report.current_consensus)

    if report.active_debates:
        st.subheader("⚖️ Active Debates")
        for d in report.active_debates:
            with st.expander(d.question):
                for i, p in enumerate(d.perspectives):
                    st.markdown(f"**Perspective {i + 1}:** {p.viewpoint}")
                    st.markdown(f"_Evidence:_ {p.evidence}")
                    if i < len(d.perspectives) - 1:
                        st.markdown("---")

    if report.notable_findings:
        st.subheader("🔬 Notable Findings")
        for f in report.notable_findings:
            year = f" ({f.year})" if f.year else ""
            st.markdown(f"- {f.finding}{year} — [source]({f.source_url})")

    if report.recent_developments:
        st.subheader("📰 Recent Developments")
        for n in report.recent_developments:
            label = n.headline + (f"  ({n.date})" if n.date else "")
            with st.expander(label):
                st.markdown(n.summary)
                if n.source_url:
                    st.markdown(f"[Open source]({n.source_url})")


def render_comparison(report: ComparisonReport) -> None:
    st.header(f"⚖️ {report.item_a}  vs  {report.item_b}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", report.comparison_category or "—")
    c2.metric("Iterations", report.research_iterations)
    c3.metric("Confidence", f"{report.confidence_score:.0%}")

    if report.summary:
        st.subheader("📋 Summary")
        st.markdown(report.summary)

    if report.dimensions:
        st.subheader("📊 Side-by-side")
        for d in report.dimensions:
            winner_tag = ""
            if d.winner:
                winner_tag = {
                    "a": f" → **{report.item_a}** wins",
                    "b": f" → **{report.item_b}** wins",
                    "tie": " → tie",
                    "depends": " → depends",
                }.get(d.winner.lower(), "")
            with st.expander(f"{d.dimension}{winner_tag}"):
                cA, cB = st.columns(2)
                cA.markdown(f"**{report.item_a}**\n\n{d.item_a_position}")
                cB.markdown(f"**{report.item_b}**\n\n{d.item_b_position}")

    if report.when_to_choose_a or report.when_to_choose_b:
        st.subheader("🎯 When to choose which")
        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"**Choose {report.item_a} when:**")
            for x in report.when_to_choose_a:
                st.markdown(f"- {x}")
        with cB:
            st.markdown(f"**Choose {report.item_b} when:**")
            for x in report.when_to_choose_b:
                st.markdown(f"- {x}")

    if report.common_misconceptions:
        st.subheader("❌ Common Misconceptions")
        for m in report.common_misconceptions:
            st.markdown(f"- {m}")

    if report.verdict:
        st.subheader("⚖️ Verdict")
        st.info(report.verdict)


def render_refusal(refusal: RefusalReport) -> None:
    st.warning("This query is outside the agent's research scope.")
    st.markdown(f"**Why:** {refusal.reason}")
    st.markdown(f"**What you can try:** {refusal.suggestion}")


# ───────────────────────────── Main ─────────────────────────────

st.title("🔍 Multi-Domain Research Agent")
st.caption("Autonomous router + multi-tool research. Built with LangGraph + Pydantic + Claude.")

query = st.text_input(
    "Research query",
    value=st.session_state.get("query", ""),
    placeholder="A company, a topic, or a comparison…",
)

col1, _ = st.columns([1, 4])
with col1:
    run_btn = st.button("🚀 Research", type="primary", use_container_width=True)

if run_btn and query.strip():
    st.subheader("🔄 Agent Activity")
    log_placeholder = st.empty()

    with st.spinner("Agent is working…"):
        app = build_graph()
        initial_state = {"query": query.strip(), "logs": []}
        final_state = None
        for chunk in app.stream(initial_state, stream_mode="values"):
            final_state = chunk
            logs = chunk.get("logs", [])
            if logs:
                log_placeholder.code("\n".join(logs), language="text")

    report = (final_state or {}).get("final_report")
    if report is None:
        st.error("Agent did not return a final report. Check logs above.")
    elif isinstance(report, RefusalReport):
        st.subheader("Result")
        render_refusal(report)
    else:
        qtype = (final_state or {}).get("query_type")
        iterations = (final_state or {}).get("iteration", 0)
        st.success(
            f"✅ Routed to **{getattr(qtype, 'value', '?')}** pipeline · "
            f"completed in {iterations} iteration(s)"
        )

        tab_view, tab_json, tab_sources = st.tabs(
            ["📊 Report", "📋 Structured JSON", "📚 Sources"]
        )

        with tab_view:
            if isinstance(report, CompanyBrief):
                render_company(report)
            elif isinstance(report, TopicReport):
                render_topic(report)
            elif isinstance(report, ComparisonReport):
                render_comparison(report)

        with tab_json:
            st.json(report.model_dump())
            label = getattr(report, "company_name", None) \
                or getattr(report, "topic", None) \
                or (f"{report.item_a}_vs_{report.item_b}" if isinstance(report, ComparisonReport) else "report")
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(report.model_dump(), indent=2),
                file_name=f"{str(label).replace(' ', '_')}_report.json",
                mime="application/json",
            )

        with tab_sources:
            sources = getattr(report, "sources", [])
            if sources:
                for src in sources:
                    st.markdown(f"- {src}")
            else:
                st.info("No sources captured.")

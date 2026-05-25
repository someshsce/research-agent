"""Pydantic schemas: routing decision, 3 report types, and LangGraph state."""
from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict, Union

from pydantic import BaseModel, Field


# ───────────────────────────── Routing ─────────────────────────────


class QueryType(str, Enum):
    COMPANY = "company"
    TOPIC = "topic"
    COMPARISON = "comparison"
    UNSUPPORTED = "unsupported"


class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    WIKIPEDIA = "wikipedia"
    NEWS_SEARCH = "news_search"


class ResearchTask(BaseModel):
    tool: ToolName
    query: str
    rationale: str = Field(description="Why this tool/query was chosen")


class RoutingDecision(BaseModel):
    """Output of the router node."""
    query_type: QueryType
    reasoning: str
    # entity slots — only the relevant ones are populated per query_type
    company_name: Optional[str] = None
    topic: Optional[str] = None
    item_a: Optional[str] = None
    item_b: Optional[str] = None


# ───────────────────────────── Shared sub-types ─────────────────────────────


class NewsItem(BaseModel):
    headline: str
    date: Optional[str] = None
    summary: str
    source_url: str


# ───────────────────────────── Company report ─────────────────────────────


class FundingInfo(BaseModel):
    total_raised: Optional[str] = None
    latest_round: Optional[str] = None
    notable_investors: list[str] = Field(default_factory=list)
    valuation: Optional[str] = None


class SWOTAnalysis(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)


class CompanyBrief(BaseModel):
    report_type: str = "company"
    company_name: str
    industry: str
    founded: Optional[int] = None
    headquarters: Optional[str] = None
    description: str
    key_products: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    recent_news: list[NewsItem] = Field(default_factory=list)
    funding: FundingInfo = Field(default_factory=FundingInfo)
    swot: SWOTAnalysis = Field(default_factory=SWOTAnalysis)
    sources: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    research_iterations: int = 0


# ───────────────────────────── Topic report ─────────────────────────────


class KeyConcept(BaseModel):
    name: str
    explanation: str


class Perspective(BaseModel):
    viewpoint: str
    evidence: str


class Debate(BaseModel):
    question: str
    perspectives: list[Perspective] = Field(default_factory=list)


class Finding(BaseModel):
    finding: str
    source_url: str
    year: Optional[int] = None


class TopicReport(BaseModel):
    report_type: str = "topic"
    topic: str
    domain: str = Field(description="e.g., 'Health Science', 'Climate', 'Technology'")
    executive_summary: str
    key_concepts: list[KeyConcept] = Field(default_factory=list)
    current_consensus: str = ""
    active_debates: list[Debate] = Field(default_factory=list)
    notable_findings: list[Finding] = Field(default_factory=list)
    recent_developments: list[NewsItem] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    research_iterations: int = 0


# ───────────────────────────── Comparison report ─────────────────────────────


class ComparisonDimension(BaseModel):
    dimension: str
    item_a_position: str
    item_b_position: str
    winner: Optional[str] = Field(
        default=None,
        description="'a', 'b', 'tie', or 'depends' — use 'depends' liberally",
    )


class ComparisonReport(BaseModel):
    report_type: str = "comparison"
    item_a: str
    item_b: str
    comparison_category: str = Field(description="e.g., 'Web frameworks', 'Diets', 'Payment processors'")
    summary: str
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
    when_to_choose_a: list[str] = Field(default_factory=list)
    when_to_choose_b: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    verdict: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    research_iterations: int = 0


# ───────────────────────────── Refusal (for unsupported) ─────────────────────────────


class RefusalReport(BaseModel):
    report_type: str = "refusal"
    reason: str
    suggestion: str


AnyReport = Union[CompanyBrief, TopicReport, ComparisonReport, RefusalReport]


# ───────────────────────────── LangGraph state ─────────────────────────────


class AgentState(TypedDict, total=False):
    query: str
    query_type: QueryType
    routing: RoutingDecision
    plan: list[ResearchTask]
    completed_tasks: list[dict]
    iteration: int
    max_iterations: int
    needs_more_research: bool
    final_report: Optional[AnyReport]
    logs: list[str]

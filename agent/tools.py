"""Tool wrappers used by the executor node."""
from __future__ import annotations

import os
from typing import Any

import wikipedia
from tavily import TavilyClient


class ResearchTools:
    """Encapsulates external research tools the agent can call."""

    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY env var is required. Get a free key at https://tavily.com"
            )
        self.tavily = TavilyClient(api_key=api_key)

    def web_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """General-purpose web search."""
        try:
            results = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            return {
                "tool": "web_search",
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    }
                    for r in results.get("results", [])
                ],
            }
        except Exception as e:  # noqa: BLE001
            return {"tool": "web_search", "query": query, "error": str(e), "results": []}

    def news_search(self, query: str, max_results: int = 5, days: int = 90) -> dict[str, Any]:
        """Recent news search via Tavily's news topic."""
        try:
            results = self.tavily.search(
                query=query,
                topic="news",
                max_results=max_results,
                days=days,
            )
            return {
                "tool": "news_search",
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "published_date": r.get("published_date", ""),
                    }
                    for r in results.get("results", [])
                ],
            }
        except Exception as e:  # noqa: BLE001
            return {"tool": "news_search", "query": query, "error": str(e), "results": []}

    def wikipedia_search(self, query: str) -> dict[str, Any]:
        """Structured background lookup via Wikipedia."""
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(query, sentences=10, auto_suggest=True)
            return {
                "tool": "wikipedia",
                "query": query,
                "results": [
                    {
                        "title": page.title,
                        "url": page.url,
                        "summary": summary,
                        "content": page.content[:3000],
                    }
                ],
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option as a fallback
            try:
                first = e.options[0]
                page = wikipedia.page(first, auto_suggest=False)
                summary = wikipedia.summary(first, sentences=10, auto_suggest=False)
                return {
                    "tool": "wikipedia",
                    "query": query,
                    "results": [
                        {
                            "title": page.title,
                            "url": page.url,
                            "summary": summary,
                            "content": page.content[:3000],
                        }
                    ],
                }
            except Exception as e2:  # noqa: BLE001
                return {"tool": "wikipedia", "query": query, "error": str(e2), "results": []}
        except Exception as e:  # noqa: BLE001
            return {"tool": "wikipedia", "query": query, "error": str(e), "results": []}

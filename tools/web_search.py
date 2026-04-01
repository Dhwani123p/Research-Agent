"""Web search tool using Tavily API."""

import os
from typing import Any
from tavily import TavilyClient


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web for a query. Returns a list of results with title, url, and content.
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        })
    return results


# Tool schema for Claude API tool use
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for up-to-date information on any topic. "
        "Returns titles, URLs, and content snippets from relevant pages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10). Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

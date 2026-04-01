"""ArXiv academic paper search tool."""

import arxiv


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search ArXiv for academic papers matching the query.
    Returns title, authors, summary, PDF URL, and published date.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],
            "summary": paper.summary[:600],
            "pdf_url": paper.pdf_url,
            "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            "arxiv_id": paper.entry_id,
        })
    return results


# Tool schema for Claude API tool use
ARXIV_SEARCH_TOOL = {
    "name": "arxiv_search",
    "description": (
        "Search ArXiv for academic papers and research on a topic. "
        "Best for scientific, technical, or academic research queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The academic topic or paper title to search for.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of papers to return (1-10). Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

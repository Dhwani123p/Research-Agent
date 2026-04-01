"""Webpage fetching and content extraction tool."""

import re
import requests
from bs4 import BeautifulSoup


def fetch_page(url: str, max_chars: int = 8000) -> dict:
    """
    Fetch a webpage and extract its main text content.
    Returns title and cleaned body text (truncated to max_chars).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"url": url, "title": "", "content": f"Failed to fetch page: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "advertisement"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else ""

    # Prefer main content containers
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.body
    )
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines)[:max_chars]

    return {"url": url, "title": title, "content": content}


# Tool schema for Claude API tool use
FETCH_PAGE_TOOL = {
    "name": "fetch_page",
    "description": (
        "Fetch the full text content of a webpage given its URL. "
        "Use this after web_search to read the actual page content in detail."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the webpage to fetch.",
            },
        },
        "required": ["url"],
    },
}

"""PDF reading tool — extract text from local file or URL."""

import io
import requests
import PyPDF2


def read_pdf(source: str, max_chars: int = 10000) -> dict:
    """
    Extract text from a PDF given a file path or URL.
    Returns the extracted text (truncated to max_chars).
    """
    try:
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=20)
            resp.raise_for_status()
            file_obj = io.BytesIO(resp.content)
        else:
            file_obj = open(source, "rb")

        reader = PyPDF2.PdfReader(file_obj)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)[:max_chars]
        return {
            "source": source,
            "pages": len(reader.pages),
            "content": full_text,
        }
    except Exception as e:
        return {"source": source, "pages": 0, "content": f"Failed to read PDF: {e}"}


# Tool schema for Claude API tool use
PDF_READER_TOOL = {
    "name": "read_pdf",
    "description": (
        "Extract and read text from a PDF file. "
        "Accepts either a local file path or a direct PDF URL (e.g. ArXiv PDF links)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or URL pointing to a PDF document.",
            },
        },
        "required": ["source"],
    },
}

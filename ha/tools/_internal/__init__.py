import requests
from typing import Dict, Any

def _post_crawl(crawler_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(crawler_url, json=payload, timeout=(5, max(10, payload.get("timeoutSecs", 30) + 5)))
    r.raise_for_status()
    return r.json()

def _shape_pages_for_llm(result: Dict[str, Any], *, mode: str = "crawl") -> str:
    """
    mode='crawl'  -> summaries + link samples
    mode='fetch'  -> fuller text for single page
    """
    pages = result.get("pages", []) or []
    lines: list[str] = []
    lines.append(f"startedAt: {result.get('startedAt')}")
    lines.append(f"count: {result.get('count', len(pages))}")
    lines.append("")

    # hard caps
    MAX_PAGES = 10 if mode == "crawl" else 1
    MAX_HEADINGS = 30
    MAX_LINKS = 40
    PREVIEW_CHARS = 2000
    FULL_CHARS = 25000

    for i, p in enumerate(pages[:MAX_PAGES], start=1):
        url = p.get("url")
        status = p.get("status")
        title = p.get("title")
        h1 = p.get("h1")
        meta = p.get("metaDescription")
        headings = (p.get("headings") or [])[:MAX_HEADINGS]
        links = (p.get("links") or [])[:MAX_LINKS]
        text = p.get("text") or ""

        lines.append(f"[{i}] {url}")
        lines.append(f"status: {status}")
        if title: lines.append(f"title: {title}")
        if h1: lines.append(f"h1: {h1}")
        if meta: lines.append(f"metaDescription: {meta}")

        if headings:
            lines.append("headings:")
            for h in headings:
                lines.append(f"  - {h}")

        if mode == "crawl":
            # preview only
            preview = text[:PREVIEW_CHARS]
            if preview:
                lines.append("textPreview:")
                lines.append(preview)
                if len(text) > PREVIEW_CHARS:
                    lines.append("...<truncated>...")
        else:
            # fuller text for single page
            body = text[:FULL_CHARS]
            if body:
                lines.append("text:")
                lines.append(body)
                if len(text) > FULL_CHARS:
                    lines.append("...<truncated>...")

        if links:
            lines.append("links:")
            for l in links:
                lines.append(f"  - {l}")

        lines.append("")

    return "\n".join(lines).strip()


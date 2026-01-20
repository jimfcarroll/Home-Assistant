from ._internal import _post_crawl, _shape_pages_for_llm
from ddgs import DDGS

from config import DEBUG, CRAWLER_URL

# -----------------------------
# Tools (ADK uses normal Python callables; docstrings matter)
# -----------------------------

def web_search(query: str) -> str:
    """
    Search the internet for current facts and information.

    Guidance:
    - Format your query starting with the intent first, then the context.
    - You will get a list of URLs back.
    - Use the URLs to read web pages with `read_web_page`.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

    ret = "".join(f"\n{r['href']}" for r in results)

    if DEBUG:
        print("\n======== web search ==========")
        print(f"query:{query}")
        print("========")
        print(ret)
        print("==============================", flush=True)

    return ret


def read_web_page(url: str) -> str:
    """
    Render a web page locally and return LLM-suitable visible text.
    """
    payload = {
        "startUrls": [url],
        "maxPages": 1,
        "maxDepth": 0,
        "sameOriginOnly": True,
        "waitUntil": "domcontentloaded",
        "includeLinks": False,
        "includeHtml": False,
        "timeoutSecs": 30,
    }
    result = _post_crawl(CRAWLER_URL, payload)
    res = _shape_pages_for_llm(result, mode="fetch")

    if DEBUG:
        print("\n======== page read ==========")
        print(res[:5000])  # preview only
        print("==============================", flush=True)

    return res


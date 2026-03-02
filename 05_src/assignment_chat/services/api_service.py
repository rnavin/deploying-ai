import requests

def wiki_summary_service(user_message: str) -> str:
    """
    Usage: 'api: topic'
    Fetches Wikipedia summary and rewrites into a friendly response.
    """
    q = user_message.strip()
    if ":" in q:
        q = q.split(":", 1)[1].strip()
    if not q:
        return "Try: `api: mediterranean diet`"

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(q)}"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "DietitianBot/1.0"})
        if r.status_code != 200:
            return f"I couldn't fetch that topic from Wikipedia (HTTP {r.status_code}). Try another query."
        data = r.json()
    except requests.RequestException as e:
        return f"Network error calling Wikipedia: {e}"

    title = data.get("title", q)
    extract = (data.get("extract") or "").strip()
    page = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    if not extract:
        return f"I found a page for **{title}**, but there was no summary available."

    return f"**{title}**\n\n{extract}\n\nSource: {page}"
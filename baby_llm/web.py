"""Utilities for scraping web pages into plain text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser


@dataclass
class WebScrapeResult:
    url: str
    title: Optional[str]
    text: str
    characters: int


class WebScrapeError(RuntimeError):
    pass


DEFAULT_HEADERS = {
    "User-Agent": "BabyLLM/1.0 (+https://example.com)",
}


def fetch_url_text(
    url: str,
    *,
    timeout: int = 15,
    max_chars: int = 20000,
) -> WebScrapeResult:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    except requests.RequestException as exc:
        raise WebScrapeError(f"Failed to fetch {url}: {exc}") from exc

    if response.status_code >= 400:
        raise WebScrapeError(f"{url} returned status {response.status_code}")

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:  # pragma: no cover
        raise WebScrapeError(f"Failed to parse HTML from {url}: {exc}") from exc

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text_chunks = list(soup.stripped_strings)
    if not text_chunks:
        raise WebScrapeError(f"No readable text detected at {url}")

    text = " ".join(text_chunks)
    if len(text) > max_chars:
        text = text[:max_chars]

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return WebScrapeResult(
        url=url,
        title=title,
        text=text,
        characters=len(text),
    )


def crawl_domain(
    root_url: str,
    *,
    max_pages: int = 5,
    max_depth: int = 1,
    max_chars: int = 20000,
    timeout: int = 15,
    infinite: bool = False,
) -> List[WebScrapeResult]:
    parsed_root = urlparse(root_url)
    if not parsed_root.scheme or not parsed_root.netloc:
        raise WebScrapeError("Provide a full URL including scheme (https://...)")

    robots_url = f"{parsed_root.scheme}://{parsed_root.netloc}/robots.txt"
    robots = RobotFileParser()
    robots.set_url(robots_url)
    try:
        robots.read()
    except Exception:
        pass

    user_agent = DEFAULT_HEADERS["User-Agent"]
    queue: List[Tuple[str, int]] = [(root_url, 0)]
    seen: Set[str] = set()
    results: List[WebScrapeResult] = []

    while queue and (len(results) < max_pages or infinite):
        current_url, depth = queue.pop(0)
        parsed = urlparse(current_url)
        if parsed.netloc != parsed_root.netloc:
            continue
        if current_url in seen:
            continue
        seen.add(current_url)

        if robots and robots.default_entry is not None:
            if not robots.can_fetch(user_agent, current_url):
                continue

        try:
            result = fetch_url_text(
                current_url, timeout=timeout, max_chars=max_chars
            )
            results.append(result)
            if not infinite and len(results) >= max_pages:
                break
        except WebScrapeError:
            continue

        if depth >= max_depth:
            continue

        try:
            page = requests.get(
                current_url, headers=DEFAULT_HEADERS, timeout=timeout
            )
        except requests.RequestException:
            continue

        soup = BeautifulSoup(page.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = urljoin(current_url, link["href"])
            parsed_href = urlparse(href)
            if parsed_href.scheme not in {"http", "https"}:
                continue
            if parsed_href.netloc != parsed_root.netloc:
                continue
            queue.append((href, depth + 1))

    if not results:
        raise WebScrapeError("Crawler did not find any readable pages.")
    return results

"""Web fetch providers — pluggable backends for the web_fetch ToolCall."""

import logging
from abc import ABC, abstractmethod

import aiohttp
from redbot.core import commands
from redbot.core.bot import Red
from trafilatura import extract

logger = logging.getLogger("red.bz_cogs.aiuser")

# ─── Provider Registry ─────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type["WebFetchProvider"]] = {}


def register_fetch_provider(name: str):
    """Decorator to register a WebFetchProvider implementation."""

    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls

    return decorator


def get_fetch_provider(backend: str) -> "WebFetchProvider":
    """Look up a fetch provider by backend name."""
    return _PROVIDERS.get(backend, _PROVIDERS.get("none", NoneProvider))


# ─── ABC ───────────────────────────────────────────────────────────────────────


class WebFetchProvider(ABC):
    """Abstract base for web fetch (get contents) backends."""

    @staticmethod
    @abstractmethod
    async def fetch(urls: list[str], guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        ...


# ─── None Provider (disabled) ──────────────────────────────────────────────────


@register_fetch_provider("none")
class NoneProvider(WebFetchProvider):
    """No-op provider returned when no backend is configured."""

    @staticmethod
    async def fetch(urls: list[str], guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        return "Error: No web fetch backend has been configured. Ask the bot owner to set one with `[p]aiuser functions web_fetch_backend`."


# ─── Exa Provider ──────────────────────────────────────────────────────────────


@register_fetch_provider("exa")
class ExaFetchProvider(WebFetchProvider):
    """Fetch page contents via Exa API using the exa-py SDK."""

    @staticmethod
    async def fetch(urls: list[str], guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        api_key = (await bot.get_shared_api_tokens("exa")).get("api_key")
        if not api_key:
            return "Error: Exa API key is not configured. The bot owner needs to set it with `[p]set api exa api_key,YOUR_KEY`."

        try:
            from exa_py import Exa
        except ImportError:
            return "Error: exa-py package is not installed. The bot owner needs to install it."

        exa = Exa(api_key=api_key)

        kwargs: dict = {}
        if config.get("text", True):
            kwargs["text"] = config.get("text")
        if config.get("summary"):
            kwargs["summary"] = config.get("summary")
        if config.get("highlights"):
            kwargs["highlights"] = config.get("highlights")
        if guiding_query:
            kwargs["highlights"]["guiding_query"] = guiding_query
        if config.get("livecrawl_timeout"):
            kwargs["livecrawl_timeout"] = config.get("livecrawl_timeout")
        if config.get("max_age_hours"):
            kwargs["max_age_hours"] = config.get("max_age_hours")



        try:
            results = exa.get_contents(urls, **kwargs)
        except Exception:
            logger.exception("Exa get_contents failed")
            return "An error occurred while fetching page contents with Exa."

        if not results.results:
            return "No content could be fetched from the provided URLs."

        parts: list[str] = []
        for r in results.results:
            title = r.title or "Untitled"
            url = r.url or ""
            if getattr(r, "text", None):
                text = r.text[:3000] if len(r.text) > 3000 else r.text
                parts.append(f"## {title}\nURL: {url}\n\n{text}")
            elif getattr(r, "summary", None):
                parts.append(f"## {title}\nURL: {url}\nSummary: {r.summary}")
            else:
                parts.append(f"## {title}\nURL: {url}\n(No text content returned)")

        return "\n\n".join(parts)


# ─── Scrape Provider (trafilatura) ─────────────────────────────────────────────


@register_fetch_provider("scrape")
class ScrapeFetchProvider(WebFetchProvider):
    """Fetch page contents by directly scraping URLs with trafilatura."""

    @staticmethod
    async def fetch(urls: list[str], guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        results = []
        for url in urls:
            try:
                content = await ScrapeFetchProvider._scrape_single(url)
                results.append(content)
            except Exception:
                logger.exception(f"Failed to scrape {url}")
                results.append(f"Error fetching {url}")
        return "\n\n".join(results)

    @staticmethod
    async def _scrape_single(link: str) -> str:
        headers = {
            "Cache-Control": "no-cache",
            "Referer": "https://www.google.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(link) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    raise ValueError(f"Content type is not text/html: {content_type}")
                html_content = await response.text()
                res = "Link content: " + extract(html_content)
                if len(res) > 5000:
                    res = res[:5000] + "..."
                return res

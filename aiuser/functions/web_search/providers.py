"""Web search providers — pluggable backends for the web_search ToolCall.

Each provider is a static async method on a class implementing WebSearchProvider.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp
from redbot.core import commands
from redbot.core.bot import Red
from trafilatura import extract

from aiuser.utils.utilities import contains_youtube_link

logger = logging.getLogger("red.bz_cogs.aiuser")

SERPER_ENDPOINT = "https://google.serper.dev/search"

# ─── Provider Registry ─────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type["WebSearchProvider"]] = {}


def register_search_provider(name: str):
    """Decorator to register a WebSearchProvider implementation."""

    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls

    return decorator


def get_search_provider(backend: str) -> "WebSearchProvider":
    """Look up a search provider by backend name. Returns NoneProvider if not found."""
    return _PROVIDERS.get(backend, _PROVIDERS.get("none", NoneProvider))


# ─── ABC ───────────────────────────────────────────────────────────────────────


class WebSearchProvider(ABC):
    """Abstract base for web search backends."""

    @staticmethod
    @abstractmethod
    async def search(search_query: str, guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        ...


# ─── None Provider (disabled) ──────────────────────────────────────────────────


@register_search_provider("none")
class NoneProvider(WebSearchProvider):
    """No-op provider returned when no backend is configured."""

    @staticmethod
    async def search(search_query: str, guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        return "Error: No web search backend has been configured. Ask the bot owner to set one with `[p]aiuser functions web_search_backend`."


# ─── Exa Provider ──────────────────────────────────────────────────────────────


@register_search_provider("exa")
class ExaSearchProvider(WebSearchProvider):
    """Search via Exa API using the exa-py SDK."""

    @staticmethod
    async def search(search_query: str, guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        api_key = (await bot.get_shared_api_tokens("exa")).get("api_key")
        if not api_key:
            return "Error: Exa API key is not configured. The bot owner needs to set it with `[p]set api exa api_key,YOUR_KEY`."

        try:
            from exa_py import Exa
        except ImportError:
            return "Error: exa-py package is not installed. The bot owner needs to install it."

        exa = Exa(api_key=api_key)

        kwargs: dict = {}
        kwargs["contents"] = config.get("contents", {"highlights": True})
        kwargs["num_results"] = config.get("num_results", 10)
        kwargs["type"] = config.get("type", "auto")

        if config.get("livecrawl_timeout"):
            kwargs["livecrawl_timeout"] = config.get("livecrawl_timeout")
        if config.get("max_age_hours"):
            kwargs["max_age_hours"] = config.get("max_age_hours")

        if kwargs["contents"].get("highlights") and guiding_query:
            kwargs["contents"]["highlights"]["guiding_query"] = guiding_query

        for opt in ("include_domains", "exclude_domains", "start_published_date", "end_published_date"):
            if config.get(opt) is not None:
                kwargs[opt] = config[opt]

        try:
            results = exa.search(search_query, **kwargs)
        except Exception:
            logger.exception("Exa search failed")
            return "An error occurred while searching the web with Exa."

        if not results.results:
            return "No relevant information was found."

        parts: list[str] = []
        for r in results.results:
            title = r.title or "Untitled"
            url = r.url or ""
            if r.highlights:
                parts.append(f"## {title}\nURL: {url}\n" + "\n".join(f"- {h}" for h in r.highlights))
            elif getattr(r, "text", None):
                text = r.text[:1000]
                parts.append(f"## {title}\nURL: {url}\n{text}")
            else:
                parts.append(f"## {title}\nURL: {url}")

        return "\n\n".join(parts)


# ─── Serper Provider (Google) ──────────────────────────────────────────────────


@register_search_provider("serper")
class SerperSearchProvider(WebSearchProvider):
    """Search via Serper.dev (Google)."""

    @staticmethod
    async def search(search_query: str, guiding_query: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        api_key = (await bot.get_shared_api_tokens("serper")).get("api_key")
        if not api_key:
            return "Error: Serper.dev API key is not configured. The bot owner needs to set it with `[p]set api serper api_key,YOUR_KEY`."

        return await SerperSearchProvider._execute_serper(search_query, api_key, ctx)

    @staticmethod
    async def _execute_serper(search_query: str, api_key: str, ctx: commands.Context) -> str:
        payload = json.dumps({"q": search_query})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(SERPER_ENDPOINT, data=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception:
            logger.exception("Failed request to serper.io")
            return "An error occurred while searching Google."

        # Answer box
        answer_box = data.get("answerBox")
        if answer_box and "snippet" in answer_box:
            return f"Use the following relevant information to generate your response: {answer_box['snippet']}"

        # Organic results
        organic_results = [
            r for r in data.get("organic", [])
            if not contains_youtube_link(r.get("link", ""))
        ]
        if not organic_results:
            return "No relevant information was found using a Google search."

        first_result = organic_results[0]
        link = first_result.get("link")

        try:
            text_content = await SerperSearchProvider._scrape_page(link, search_query, ctx)
            return f"Use the following relevant information to generate your response: {text_content}"
        except Exception:
            logger.debug(f"Failed scraping URL {link}", exc_info=True)
            knowledge_graph = data.get("knowledgeGraph", {})
            if knowledge_graph:
                return f"Use the following relevant information to generate your response: {SerperSearchProvider._format_knowledge_graph(knowledge_graph)}"
            return f"Use the following relevant information to generate your response: {first_result.get('snippet', 'N/A')}"

    @staticmethod
    async def _scrape_page(link: str, search_query: str, ctx: commands.Context) -> str:
        headers = {
            "Cache-Control": "no-cache",
            "Referer": "https://www.google.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        }
        logger.info(f"Requesting {link} from Google query \"{search_query}\" in {ctx.guild.name}")
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(link) as response:
                response.raise_for_status()
                html_content = await response.text()
                text_content = extract(html_content)
                if len(text_content) > 5000:
                    text_content = text_content[:5000] + "..."
                return text_content

    @staticmethod
    def _format_knowledge_graph(knowledge_graph: dict) -> str:
        title = knowledge_graph.get("title", "")
        kg_type = knowledge_graph.get("type", "")
        description = knowledge_graph.get("description", "")
        text_content = f"{title} - ({kg_type}) \n {description}"
        attributes = knowledge_graph.get("attributes", {})
        for attribute, value in attributes.items():
            text_content += f" \n {attribute}: {value}"
        return text_content

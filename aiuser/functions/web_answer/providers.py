"""Web answer providers — pluggable backends for the web_answer ToolCall."""

import logging
from abc import ABC, abstractmethod

from redbot.core import commands
from redbot.core.bot import Red

logger = logging.getLogger("red.bz_cogs.aiuser")

# ─── Provider Registry ─────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type["WebAnswerProvider"]] = {}


def register_answer_provider(name: str):
    """Decorator to register a WebAnswerProvider implementation."""

    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls

    return decorator


def get_answer_provider(backend: str) -> "WebAnswerProvider":
    """Look up an answer provider by backend name."""
    return _PROVIDERS.get(backend, _PROVIDERS.get("none", NoneProvider))


# ─── ABC ───────────────────────────────────────────────────────────────────────


class WebAnswerProvider(ABC):
    """Abstract base for web answer backends."""

    @staticmethod
    @abstractmethod
    async def answer(question: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        ...


# ─── None Provider (disabled) ──────────────────────────────────────────────────


@register_answer_provider("none")
class NoneProvider(WebAnswerProvider):
    """No-op provider returned when no backend is configured."""

    @staticmethod
    async def answer(question: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        return "Error: No web answer backend has been configured. Ask the bot owner to set one with `[p]aiuser functions web_answer_backend`."


# ─── Exa Provider ──────────────────────────────────────────────────────────────


@register_answer_provider("exa")
class ExaAnswerProvider(WebAnswerProvider):
    """Answer questions via Exa API using the exa-py SDK."""

    @staticmethod
    async def answer(question: str, bot: Red, ctx: commands.Context, config: dict) -> str:
        api_key = (await bot.get_shared_api_tokens("exa")).get("api_key")
        if not api_key:
            return "Error: Exa API key is not configured. The bot owner needs to set it with `[p]set api exa api_key,YOUR_KEY`."

        try:
            from exa_py import Exa
        except ImportError:
            return "Error: exa-py package is not installed. The bot owner needs to install it."

        exa = Exa(api_key=api_key)

        kwargs: dict = {}
        if config.get("text"):
            kwargs["text"] = True

        try:
            response = exa.answer(question, **kwargs)
        except Exception:
            logger.exception("Exa answer failed")
            return "An error occurred while getting an answer with Exa."

        parts = [response.answer if response.answer else "No answer was returned."]

        if response.citations:
            parts.append("\nCitations:")
            for i, citation in enumerate(response.citations, 1):
                title = citation.title or citation.url
                parts.append(f"{i}. {title} <{citation.url}>")
                if getattr(citation, "text", None):
                    parts.append(f"   {citation.text[:300]}...")

        return "\n".join(parts)

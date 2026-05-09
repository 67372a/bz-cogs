# bz-cogs Documentation

This directory contains the comprehensive project research documentation for the [`bz-cogs`](https://github.com/zhaobenny/bz-cogs) repository — a collection of Red-DiscordBot cogs.

## Repository Structure

| Path | Description |
|------|-------------|
| [`info.json`](../info.json) | Repository metadata (author: `zhaobenny`) |
| [`aiuser/`](../aiuser/) | **Primary cog**: Human-like Discord interactions powered by LLMs |
| [`aiemote/`](../aiemote/) | Human-like Discord reacts to messages powered by OpenAI |
| [`aimage/`](../aimage/) | A1111 Stable Diffusion image generation in Discord |
| `oneletteronly/` | Sets nicknames to initials for new users |

## Documentation Index

| File | Covers |
|------|--------|
| [Project Overview](project-overview.md) | High-level description of the entire aiuser cog, system architecture, and core functionality |
| [Core Architecture](architecture.md) | Message processing pipeline, validation chain, trigger system, queue management, and core cog initialization |
| [LLM Pipeline](llm-pipeline.md) | Two-phase execution model, context building (MessagesList), response formatting, and delivery |
| [Function Calling](functions-tools.md) | ToolCall base class, available local tools, OpenRouter server tools, discovery mechanism |
| [Image Generation](image-generation.md) | Image request flow, provider factory, supported endpoints, direct image generation tool |
| [Configuration](configuration.md) | Settings system, config hierarchy, defaults, model support lists, owner settings |
| [Types & Utilities](types-utilities.md) | Core types, enums, caching, helper functions, type definitions |

## Key Facts

- **Red-DiscordBot minimum version**: 3.5.0
- **Dependencies**: `openai>=2.33.0`, `tenacity`, `tiktoken`, `Pillow`, `trafilatura`, `aiohttp`, `opencv-python-headless`, `numpy>=1.26.4`, `cachetools`
- **License**: MIT
- **Default LLM Model**: `google/gemini-2.5-flash`
- **Architecture Pattern**: Composite/Mixin pattern merging settings, dashboard, random messages, and core cog functionality

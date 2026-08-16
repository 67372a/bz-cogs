# `aiuser` Cog — Project Overview

## What Is It?

[`aiuser`](../aiuser/) is a Red-DiscordBot cog that provides **human-like Discord interactions** powered by LLMs (OpenAI, OpenRouter, or any OpenAI-compatible endpoint). The bot can seamlessly participate in conversations, parse multiple types of media, perform function calls, generate images, and send spontaneous messages — acting as a virtual user.

## Core Functionality

- **Chat Completions**: Listens to messages in whitelisted channels or when directly mentioned/replied to, and queries an LLM pipeline to generate a response.
- **Media Processing (Vision/Documents/Audio/Video)**: Capable of parsing various media types into LLM-friendly formats (text extraction from documents/PDFs, OCR/BLIP/LLM image scanning, OCR for stickers, etc.).
- **Function Calling**: Contains tools that the LLM can use to interact with its environment (react to messages, search the web, get weather, generate images, etc.).
- **Image Requests**: Integrates with Stable Diffusion endpoints (e.g. Automatic1111, Modal, Dall-E) to create images dynamically based on chat context.
- **Random Messages**: Capable of spontaneously talking in configured channels on a scheduled loop (every 33 minutes).
- **Opt-in / Opt-out System**: Employs a robust data privacy framework, keeping history exclusively from users who have opted in (or if `optin_by_default` is activated for a specific server).

## Author & Credits

- **Author**: `zhaobenny`
- **Contributor**: `67372a`
- **License**: MIT
- **Support Server**: [Discord](https://discord.gg/GwT2yHPqzN)

## Technical Stack

| Component | Technology |
|-----------|------------|
| Bot Framework | `Red-DiscordBot` 3.5.0+ / `discord.py` |
| LLM Client | `openai>=2.33.0` (AsyncOpenAI) |
| Tokenization | `tiktoken` |
| Retry Logic | `tenacity` |
| Web Scraping | `trafilatura` + `BeautifulSoup` |
| Image Processing | `Pillow`, `opencv-python-headless` (local OCR/BLIP) |
| HTTP | `aiohttp`, `httpx2` |
| Caching | `cachetools` (TTL-based emoji caching), custom LRU `Cache` class |

## System Architecture Diagram

```
Discord Message
    │
    ▼
on_message_without_command()
    │
    ▼
validators.py ──► 5-step validation chain
    │
    ▼
triggers.py ──► check_triggers() (mention, keywords, grok, conversation)
    │
    ▼
handlers.py ──► queue_response()
    │
    ▼
process_queue() [per-channel sequential queue, 1s buffer]
    │
    ▼
dispatch_response()
    │
    ├──► is_image_request() ──► image generator factory ──► image response
    │
    └──► create_messages_list() ──► MessagesList (context building)
              │
              ▼
         LLMPipeline.phase1() ──► first LLM call
              │
              ├──► No tools ──► send response immediately
              │
              └──► Has tools ──► phase2() ──► execute tools ──► second LLM call ──► send combined response
```

## Key Configuration File

[`aiuser/info.json`](../aiuser/info.json) registers the cog with Red-DiscordBot, declaring:
- Required permissions: `read_messages`, `send_messages`, `embed_links`, `attach_files`, `add_reactions`
- Python package requirements (listed above)
- End-user data statement clarifying data forwarding to third-party LLM providers
- Tags: `openai`, `chatgpt`, `chatbot`, `openrouter`, `ai`, `image generation`, `llm`

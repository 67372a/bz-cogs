# AI User (aiuser) Overview

`aiuser` is a Red-DiscordBot cog that provides human-like Discord interactions powered by OpenAI or compatible endpoints (e.g., OpenRouter, Local LLMs). The primary purpose of the cog is to allow a Discord bot to seamlessly participate in conversations, parse multiple types of media, and perform function calls, acting as a virtual user. 

## Core Functionality
- **Chat Completions**: Listens to messages in whitelisted channels or when directly mentioned/replied to, and queries an LLM pipeline to generate a response.
- **Media Processing (Vision/Documents/Audio/Video)**: Capable of parsing various media types into LLM-friendly formats (text extraction from documents/PDFs, OCR/BLIP/LLM image scanning, OCR for stickers, etc.).
- **Function Calling**: Contains tools that the LLM can use to interact with its environment.
- **Image Requests**: Integrates with Stable Diffusion endpoints (e.g. Automatic1111, Modal) to create images dynamically based on the chat context.
- **Random Messages**: Capable of spontaneously talking in configured channels.
- **Opt-in / Opt-out System**: Employs a robust data privacy framework, keeping history exclusively from users who have opted in (or if `optin_by_default` is activated for a specific server).

## System Architecture

### 1. **Message Handlers** (`core/handlers.py`, `core/validators.py`)
Intercepts standard Discord `on_message` events and slash commands.
- `validators.py`: Analyzes the validity of messages, permissions, blacklisted lists, missing whitelists, skipped channels, and specific sender conditions. **Importantly, we allow external bots (like Tupperbox or distinct instances) to be treated as real participants**, expanding the interaction space without confusing `aiuser`'s own persona identity.
- `handlers.py`: Queues verified messages for context-building.

### 2. **Context Creation** (`messages_list/*`)
Responsible for converting a localized Discord history into an LLM context array (ChatML).
- Messages sent by users are formatted into an XML-like structure denoting `<message... author_id="..." username="...">` to help the LLM contextualize the chat.
- Backwards reads the conversation history (configured using `messages_backread` and `messages_backread_seconds`) to create a context window, ensuring time-gaps don't disrupt continuity.
- Evaluates the proper persona format from Role/Member/Channel/Guild specifications. Prefills and system instructions are determined sequentially based on highest priority. 

### 3. **LLM Pipeline** (`response/chat/llm_pipeline.py`)
Responsible for communicating with the endpoint in an attempt to form a correct response.
- Decorates the generation with extensive `tenacity` retries, designed to recover from API connection errors or `RateLimitErrors`. It also retries unsatisfactory responses (such as empty content filtered by OpenAI's internal filters).
- Integrates function calls synchronously. When the LLM calls a function, the pipeline resolves the `tool_call` and embeds the output back into the history, prompting a new follow-up generation.

### 4. **Response Dispatcher** (`response/dispatcher.py`, `response/chat/response.py`)
Responsible for the final formatting and safe delivery of the generated text or image.
- Employs regex replacements to strip the text of Discord bots' internal mentions and variables. Wait processes guarantee the text doesn't contain the bot's raw ID.
- Uses pagination mechanisms if the textual content exceeds Discord's max 4096 output embed limitation. 
- Properly processes Reasoning (e.g., from newer models like `o3-mini`) separately and logs it into a "Thoughts" collapsible embed, to prevent thought processes from leaking into general response texts.

## Future Development & Extensibility
- **Custom Endpoints**: Endpoints can be specified by the bot owner. Future updates should continue supporting varying API structures or varying implementations of `tool_calls`.
- **Bot Interactions**: As distinct AI accounts and bots interact with people, `aiuser` continues to improve contextual alignment over extended conversations involving multiple bots.
- **Embed Filtering**: Ongoing adjustments to `.removelist_regexes()` handles complex model hallucinated tags, helping prune AI-specific prefixes (e.g. `<message .../>`) from slipping into output strings.

## Extensible Function Calling (`functions/`)
`aiuser` registers schema-driven function calls directly with the LLM. Available Tools include:

### Discord-Native Actions
- **`react_to_message(message_id, emoji, reason)`**: Allows the bot to autonomously drop a reaction emoji (including custom server emojis) to express feelings about specific past messages.
- **`pin_message(message_id, reason)`**: Lets the bot pin exceptionally important or entertaining messages to the channel history limits.
- **`send_tts_message(reason)`**: Instructs the pipeline to return the next output as a Text-to-Speech (TTS) response.
- **`change_user_nickname(username, nickname)`**: Changes the Discord nickname of a specified user (assuming proper guild hierarchy and bot permissions).
- **`timeout_user(username, reason)`**: Casts a quick symbolic timeout (3 seconds) onto a user using Discord's moderation capabilities.

### Media & Creativity
- **`generate_image(prompt)`**: Invokes the `google/gemini-2.5-flash-image-preview` model via the Open Router API to generate and send an original image based on a requested visual prompt.

### Utility & Searches
- **`open_url(url)`**: Triggers a web scraper (via `trafilatura`/`BeautifulSoup`) to fetch and summarize web content when links are introduced.
- **`search_google(query)`**: Hits the Serper API to quickly query Google for live information.
- **`wolfram_alpha(query)`**: Executes mathematical and factual queries.
- **`get_weather(location, days)` / `get_local_weather(days)` / `is_daytime_local()`**: Localized tools for returning real-world weather metrics.
- **`do_not_respond(reason, respond)`**: An emergent tool enabling the bot to specifically choose to silently ignore a conversation.

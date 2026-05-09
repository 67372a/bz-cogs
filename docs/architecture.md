# Core Architecture

## 1. Cog Class & Initialization

The main entry point is [`AIUser`](../aiuser/core/aiuser.py:33), which uses a **CompositeMetaClass** pattern to merge multiple mixin classes:

```python
class AIUser(DashboardIntegration, Settings, RandomMessageTask, commands.Cog, metaclass=CompositeMetaClass):
```

### Composite Class Hierarchy

| Mixin | File | Purpose |
|-------|------|---------|
| `DashboardIntegration` | [`dashboard/base.py`](../aiuser/dashboard/base.py:10) | Web dashboard pages (consent, main config, owner config) |
| `Settings` | [`settings/base.py`](../aiuser/settings/base.py:33) | All `[p]aiuser` command groups (9 sub-mixins) |
| `RandomMessageTask` | [`core/random_message_task.py`](../aiuser/core/random_message_task.py:17) | Periodic random message task loop |
| `commands.Cog` | Red core | Base Red-DiscordBot cog functionality |

### Abstract Base: `MixinMeta`

[`MixinMeta`](../aiuser/types/abc.py:26) defines the shared interface contract that all mixins expect:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.bot` | `Red` | Red bot instance |
| `self.config` | `Config` | Red config system |
| `self.openai_client` | `AsyncOpenAI` | LLM API client |
| `self.channels_whitelist` | `dict[int, list[int]]` | Guild → whitelisted channel IDs |
| `self.cached_messages` | `Cache[int, MessageEntry]` | LRU message conversion cache |
| `self.message_queues` | `dict[int, Queue]` | Per-channel response queues |
| `self.processing_tasks` | `dict[int, Task]` | Active queue processor tasks |
| `self.backfill_anchors` | `dict[int, Message]` | One-shot backfill anchors |
| `self.optindefault` | `dict[int, bool]` | Guild opt-in-by-default flag |
| `self.ignore_regex` | `dict[int, Pattern]` | Guild ignore regex |
| `self.override_prompt_start_time` | `dict[int, datetime]` | "Forget" cutoff |

### Initialization Flow ([`cog_load()`](../aiuser/core/aiuser.py:68))

1. Creates `AsyncOpenAI` client via [`setup_openai_client()`](../aiuser/core/openai_utils.py:18)
2. Loads guild configs into caches (`optindefault`, `channels_whitelist`, `ignore_regex`)
3. Starts the random message loop (`random_message_trigger.start()`)
4. Registers 5 config scopes: `Member`, `Role`, `Channel`, `Guild`, `Global`

---

## 2. Message Validation Chain

[`is_valid_message()`](../aiuser/core/validators.py:13) runs 5 sequential validation checks:

| Step | Validator | Purpose |
|------|-----------|---------|
| 1 | [`check_openai_client()`](../aiuser/core/validators.py:40) | Ensure LLM client exists or can be (re)created |
| 2 | [`check_guild_permissions()`](../aiuser/core/validators.py:49) | Not in DMs, cog not disabled, channel/guild not ignored |
| 3 | [`check_channel_settings()`](../aiuser/core/validators.py:67) | Channel (or parent thread) is in whitelist |
| 4 | [`check_user_status()`](../aiuser/core/validators.py:85) | Not bot self, whitelist/blacklist pass, opted in (or by-default), role/member whitelist |
| 5 | [`check_message_content()`](../aiuser/core/validators.py:113) | No lone mention, meets min length, no ignore regex match |

Each validator returns `(is_valid: bool, reason: str)`. Critical failures (OpenAI client) are logged as warnings.

---

## 3. Trigger System

[`check_triggers()`](../aiuser/core/triggers.py:56) evaluates multiple triggers and short-circuits on the first `True`:

| Trigger | File:Line | Condition |
|---------|-----------|-----------|
| [`is_bot_mentioned_or_replied()`](../aiuser/core/validators.py:130) | Bot is in message mentions AND `reply_to_mentions_replies` is enabled |
| [`is_always_reply_on_words_triggered()`](../aiuser/core/triggers.py:47) | Message contains any word from `always_reply_on_words` guild config |
| [`is_grok_triggered()`](../aiuser/core/triggers.py:35) | Message ≤25 words, contains a primary trigger ("grok"/"gork") AND secondary trigger ("true"/"explain"/"confirm"), and `grok_trigger` is enabled |
| [`is_in_conversation()`](../aiuser/core/triggers.py:15) | Recent bot message in channel (within `conversation_reply_time` seconds), evaluated with `conversation_reply_percent` probability |

If no trigger fires, falls through to RNG-based reply percentage check via [`get_percentage()`](../aiuser/core/handlers.py:90).

---

## 4. Reply Percentage Hierarchy

[`get_percentage()`](../aiuser/core/handlers.py:90) resolves in cascading priority:

1. **Member** override → 2. **Role** override (highest role) → 3. **Channel** override → 4. **Guild** default → 5. **Global default** (`1%`)

---

## 5. Queue System

[`queue_response()`](../aiuser/core/aiuser.py:122) and [`process_queue()`](../aiuser/core/aiuser.py:137):

- Creates a per-channel `asyncio.Queue` if one doesn't exist
- Puts `(ctx, messages_list)` tuples into the queue
- If no processor task is running for the channel, spawns one
- Processor drains the queue sequentially with a **1-second buffer** between messages
- Rate-limit check: if `ratelimit_reset` timestamp is in the future, responds with 💤 reaction and returns

---

## 6. Slash Command

[`slash_command()`](../aiuser/core/aiuser.py:105) (`/chat`):
- Cooldown: 1 per 30s (global) + 1 per 5s (per user)
- Input range: 1–2000 characters
- Routed through [`handle_slash_command()`](../aiuser/core/handlers.py:21) which validates and queues

---

## 7. OpenAI Client Setup

[`setup_openai_client()`](../aiuser/core/openai_utils.py:18) selects the API configuration based on the configured endpoint:

| Endpoint Type | API Key Source | Special Headers |
|---------------|---------------|-----------------|
| OpenAI (`https://api.openai.com/`) | `[p]set api openai` | None |
| OpenRouter (`https://openrouter.ai/api/`) | `[p]set api openrouter` | `HTTP-Referer` + `X-Title` |
| Custom (any other URL) | `[p]set api openai` (or placeholder key) | None |

**Rate-limit tracking**: [`create_ratelimit_hook()`](../aiuser/core/openai_utils.py:112) intercepts OpenAI HTTP responses, parses `x-ratelimit-remaining-*` headers, and writes a locked-out timestamp to config when limits are hit. Only applies to OpenAI endpoints.

---

## 8. Random Message Task

[`RandomMessageTask`](../aiuser/core/random_message_task.py:17) runs every **33 minutes**:

1. Iterates all guilds → picks random whitelisted channel
2. Validates: cog enabled, channel not ignored, random messages toggle on, RNG passes `random_messages_percent`
3. Checks: last message was >1 hour ago AND not by the bot
4. Picks random topic from `random_messages_prompts`
5. Builds minimal `MessagesList` (no history) with topic injected as system instruction
6. Queues for response

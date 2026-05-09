# Image Generation System

## 1. Image Request Detection

When a message is received, [`dispatch_response()`](../aiuser/response/dispatcher.py:16) first checks if the user is requesting an image:

1. **Trigger Words**: Messages containing words from `image_requests_trigger_words` (default: "image", "picture", "photo", etc.) AND `image_requests_second_person_trigger_words` ("yourself", "you")
2. **LLM Classification** ([`is_image_request()`](../aiuser/response/is_image_request.py)): Uses an LLM prompt: *"Your role is to identify whether each specific message is asking you to send a picture of yourself or not..."*

If an image request is detected, the dispatcher routes to the image generation system.

---

## 2. Provider Factory

[`get_image_generator()`](../aiuser/response/image/generator_factory.py:17) selects the image provider based on the configured endpoint URL:

| Condition | Provider | File |
|-----------|----------|------|
| Starts with `"dall-e-"` | `DalleImageGenerator` | [`providers/dalle.py`](../aiuser/response/image/providers/dalle.py) |
| Starts with `"https://waifus-api.nemusona.com/"` | `NemusonaGenerator` | [`providers/nemusona.py`](../aiuser/response/image/providers/nemusona.py) |
| Starts with `"https://perchance.org/ai-text-to-image-generator"` | `PerchanceGenerator` | [`providers/perchance.py`](../aiuser/response/image/providers/perchance.py) |
| Ends with `"imggen.modal.run/"` | `ModalImageGenerator` | [`providers/modal.py`](../aiuser/response/image/providers/modal.py) |
| Starts with `NINETEEN_API_URL` | `NineteenGenerator` | [`providers/nineteen.py`](../aiuser/response/image/providers/nineteen.py) |
| Starts with `"https://api.runpod.ai/v2/"` | `RunPodGenerator` | [`providers/runpod.py`](../aiuser/response/image/providers/runpod.py) |
| Starts with `IMAGE_REQUEST_AIHORDE_URL` | `AIHordeGenerator` | [`providers/aihorde.py`](../aiuser/response/image/providers/aihorde.py) |
| Any other | `GenericImageGenerator` | [`providers/generic.py`](../aiuser/response/image/providers/generic.py) |

The base class [`ImageGenerator`](../aiuser/response/image/providers/generator.py) defines the interface.

### Provider Configuration

| Provider | API Key Config | Notes |
|----------|---------------|-------|
| **Dall-E** | `[p]set api openai api_key,...` | Native OpenAI image generation |
| **Modal** | `[p]set api modal-img-gen token,...` | Serverless SD endpoint; template at [serverless-img-gen](https://github.com/zhaobenny/serverless-img-gen) |
| **RunPod** | `[p]set api runpod apikey,...` | Uses `/runsync` endpoint; [template](https://github.com/ashleykleynhans/runpod-worker-a1111/) |
| **AI Horde** | `[p]set api aihorde apikey,...` | Crowdsourced distributed cluster |

---

## 3. SD Prompt Generation

[`DEFAULT_IMAGE_REQUEST_SD_GEN_PROMPT`](../aiuser/config/defaults.py:21) instructs the LLM to act as a Stable Diffusion prompt generator. The formula:

> `[Subject] + [Emotions] + [Verb] + [Adjectives] + [Environment] + [Lighting] + [Photography type] + [Quality]`

Sample output: *"Beautiful woman, contemplative and reflective, sitting on a bench, cozy sweater, autumn park with colorful leaves, soft overcast light, muted color photography style, 4K quality."*

---

## 4. Direct Image Generation Tool (Local)

[`GenerateImageToolCall`](../aiuser/functions/generate_image/tool_call.py:44) is a **local** `ToolCall` that:
- Makes API calls directly from the bot using the existing `AsyncOpenAI` client
- Does NOT rely on OpenRouter server-side execution
- Accepts: `prompt` (detailed narrative), `aspect_ratio` (14 options), optional `reference_messages`
- Configurable model (e.g., `google/gemini-2.5-flash-image-preview`) via `direct_image_generation_parameters`
- Supports reference images: extracts from Discord message links, downloads (max 20MB, 14 images), and includes in the API call
- Caches generated images in [`image_cache`](../aiuser/utils/image_cache.py)
- Stores results in `self.generated_images` for the pipeline to send alongside text

### Image Editing

[`edit_image`](../aiuser/functions/edit_image/tool_call.py) tool allows the LLM to edit existing images (download → edit → re-encode).

---

## 5. OpenRouter Server-Side Image Generation

[`OpenRouterImageGeneration`](../aiuser/functions/openrouter/image_generation.py:21) is a **server-side tool** handled entirely by OpenRouter:

- Configured via `openrouter_image_generation_parameters` (model, quality, size, aspect_ratio, etc.)
- When called, OpenRouter processes the image generation server-side
- Response text contains an `imageUrl` extracted via regex strategies:
  1. URLs with image file extensions (.png, .jpg, .webp, etc.)
  2. Generic CDN/asset URLs (contains `/cdn/`, `/storage/`, `/generated/`, etc.)
- Images are downloaded and sent as Discord embeds

### Transparent Mode

If `openrouter_image_generation_parameters.transparent_execution` is set, the tool image content is returned directly in the LLM response without requiring explicit `tool_calls`. The pipeline detects this in [`phase2()`](../aiuser/response/chat/llm_pipeline.py:560) and processes accordingly.

---

## 6. Image Scanning (Message Content)

Three modes for scanning images in messages:

| Mode | Backend | File |
|------|---------|------|
| **LLM** (`supported-llm`) | Uses a vision-capable LLM model to analyze images | Set via `scan_images_model` |
| **AI Horde** (`ai-horde`) | Crowdsourced Image Alchemy captioning | [`converter/image/AI_horde.py`](../aiuser/messages_list/converter/image/AI_horde.py) |
| **Local** (`local`) | Tesseract OCR + BLIP captioning (CPU intensive) | [`converter/image/local.py`](../aiuser/messages_list/converter/image/local.py) |

Upload limits (configurable per guild):
- **Images**: 25 MB
- **Videos**: 10 MB
- **Documents**: 10 MB
- **Audio**: 10 MB

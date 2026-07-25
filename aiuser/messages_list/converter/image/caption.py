import base64
import logging
from io import BytesIO

from discord import Message
from PIL import Image
import cv2
import numpy as np

from aiuser.types.abc import MixinMeta
from aiuser.types.enums import ScanImageMode
from aiuser.messages_list.converter.helpers import format_text_content, _get_msg_header
from aiuser.messages_list.converter.image.AI_horde import \
    process_image_ai_horde
from aiuser.utils.image_cache import image_cache, processed_image_cache
from aiuser.utils.image_processing import (
    compute_byte_hash,
    compute_pixel_hash,
    process_image_for_llm,
    build_webp_data_url,
)

logger = logging.getLogger("red.bz_cogs.aiuser")


async def _fetch_attachment_bytes(attachment) -> bytes:
    """Download attachment bytes, using the URL-level cache to avoid re-fetches.

    Falls back to ``attachment.save()`` if the URL cache has no entry.
    """
    cached = image_cache.get(attachment.url)
    if cached is not None:
        return cached
    buffer = BytesIO()
    await attachment.save(buffer)
    raw = buffer.getvalue()
    # Store in URL cache so the same attachment URL isn't re-downloaded
    ct = attachment.content_type or "application/octet-stream"
    image_cache.set(attachment.url, raw, ct)
    return raw


def _resolve_max_pixels(mode: ScanImageMode, config_max_pixels) -> int:
    if config_max_pixels is not None:
        return config_max_pixels
    if mode == ScanImageMode.LLM:
        return 16_777_216  # 4096 * 4096
    return 1_048_576  # 1024 * 1024


def _decode_webp_to_pil(webp_bytes: bytes) -> Image.Image:
    """Decode WebP bytes back into a PIL Image (for LOCAL/AI_HORDE modes)."""
    return Image.open(BytesIO(webp_bytes))


async def _process_attachment(
    cog: MixinMeta,
    message: Message,
    attachment,
    mode: ScanImageMode,
    max_pixels: int,
):
    """Core image-attachment processing with double-keyed cache lookup.

    Returns the processed content (list of content parts for LLM mode,
    or a string for caption modes), or ``None`` on failure.
    """
    raw_bytes = await _fetch_attachment_bytes(attachment)

    # Compute hashes for the double-keyed cache
    pixel_hash = compute_pixel_hash(raw_bytes)

    # Check the processed cache first
    cached_processed = processed_image_cache.get(raw_bytes, pixel_hash, max_pixels)
    if cached_processed is not None:
        logger.debug(
            f"[ImageScan] Processed cache hit for attachment {attachment.filename} "
            f"in message {message.id}"
        )
    else:
        # Process: decode, scale, re-encode as WebP
        cached_processed = process_image_for_llm(raw_bytes, max_pixels)
        if cached_processed is None:
            logger.warning(
                f"[ImageScan] Failed to process attachment {attachment.filename} "
                f"in message {message.id}"
            )
            return None
        # Store in the double-keyed processed cache
        processed_image_cache.set(raw_bytes, pixel_hash, max_pixels, cached_processed)

    # Dispatch by mode
    if mode == ScanImageMode.LLM:
        content = []
        data_url = build_webp_data_url(cached_processed)
        content.append(
            {"type": "image_url", "image_url": {"url": data_url}}
        )
        text = format_text_content(message)
        if text:
            content.append({"type": "text", "text": text})
        else:
            # Include message metadata even when there is no text content,
            # so the LLM knows who sent the image and any reply context.
            # Bot-sent images are annotated too, for consistent provenance.
            if message.author.id == message.guild.me.id:
                content.append({"type": "text", "text": "Sent image"})
            else:
                content.append({"type": "text", "text": f'{_get_msg_header(message)}</message>'})
        return content

    elif mode == ScanImageMode.AI_HORDE:
        pil_image = _decode_webp_to_pil(cached_processed)
        return await process_image_ai_horde(cog, message, pil_image)

    elif mode == ScanImageMode.LOCAL:
        try:
            from aiuser.messages_list.converter.image.local import \
                process_image_locally
            pil_image = _decode_webp_to_pil(cached_processed)
            return await process_image_locally(cog, message, pil_image)
        except ImportError:
            logger.exception(
                "Local image scanning dependencies not installed, "
                "check cog README for instructions"
            )
            return None

    return None


async def transcribe_image(cog: MixinMeta, message: Message):
    """Transcribe the first attachment of a Discord message.

    Uses the unified image processing pipeline with double-keyed caching
    to avoid redundant downloads and reprocessing.
    """
    config = cog.config
    attachment = message.attachments[0]
    mode = ScanImageMode(await config.guild(message.guild).scan_images_mode())
    config_max_pixels = await config.guild(message.guild).max_image_pixels()
    max_pixels = _resolve_max_pixels(mode, config_max_pixels)

    content = await _process_attachment(cog, message, attachment, mode, max_pixels)

    if content:
        cache_key = f"{message.channel.id}:{message.id}"
        cog.cached_messages[cache_key] = content
        cog.cached_messages[message.id] = content

    return content


async def transcribe_single(
    cog: MixinMeta, message: Message, attachment
) -> list | str | None:
    """Transcribe a single image attachment using the unified pipeline.

    Used by ``transcribe_image_single`` in ``converter.py`` for
    multi-attachment messages.  Returns the processed content.
    """
    config = cog.config
    mode = ScanImageMode(await config.guild(message.guild).scan_images_mode())
    config_max_pixels = await config.guild(message.guild).max_image_pixels()
    max_pixels = _resolve_max_pixels(mode, config_max_pixels)

    return await _process_attachment(cog, message, attachment, mode, max_pixels)


# Keep scale_image available for any external callers that import it directly.
def scale_image(cv_image: np.ndarray, max_pixel_count: int) -> np.ndarray:
    """Scale *cv_image* down so its pixel count ≤ *max_pixel_count*.

    Preserves aspect ratio.  Returns the original if already within limits.
    """
    original_height, original_width = cv_image.shape[:2]
    original_pixel_count = original_width * original_height

    if original_pixel_count <= max_pixel_count:
        logger.debug("Image is already smaller than the max pixel count. No scaling needed.")
        return cv_image

    scale_ratio = np.sqrt(max_pixel_count / original_pixel_count)
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    scaled_image = cv2.resize(
        cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return scaled_image

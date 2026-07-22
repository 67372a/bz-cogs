import asyncio
import base64
import logging
import re
from typing import List, Optional, Tuple

import aiohttp

from aiuser.utils.image_cache import image_cache, processed_image_cache
from aiuser.utils.image_processing import (
    compute_pixel_hash,
    process_image_for_llm,
    build_webp_data_url,
)

logger = logging.getLogger("red.bz_cogs.aiuser")

# Matches URLs that point to common image formats
IMAGE_URL_PATTERN = re.compile(
    r"(https?://[^\s<>\"']+?\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|svg))(\?[^\s<>\"']*)?(?=[\s<>\"']|$)",
    re.IGNORECASE,
)

# Maximum image file size for download (20 MB default)
MAX_IMAGE_DOWNLOAD_SIZE = 20 * 1024 * 1024


class OpenRouterImageParsing:
    """Handles image URL detection, retrieval, and encoding for LLM processing.

    This is NOT a ToolCall subclass. It is a pre-processor service that:
    1. Detects image URLs in Discord message text
    2. Downloads images from URLs and base64-encodes them
    3. Builds image_url content parts for the messages array
    """

    @classmethod
    def detect_image_urls(cls, message_content: str) -> List[str]:
        """Scan message text for image URLs.

        Args:
            message_content: The raw text content of a Discord message.

        Returns:
            List of unique image URLs found in the text.
        """
        if not message_content:
            return []

        matches = IMAGE_URL_PATTERN.findall(message_content)
        urls: List[str] = []
        seen: set = set()
        for match in matches:
            # match is a tuple: (base_url, query_string_or_none)
            full_url = match[0]
            if match[1]:
                full_url += match[1]
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)

        return urls

    @classmethod
    async def fetch_image(cls, url: str, max_size: int = MAX_IMAGE_DOWNLOAD_SIZE) -> Optional[bytes]:
        """Download an image from a URL, using a global TTL cache.

        Images are cached by SHA-256 of the URL for up to 30 minutes,
        avoiding redundant downloads of the same image across messages.

        Args:
            url: The URL of the image to download.
            max_size: Maximum allowed download size in bytes.

        Returns:
            Raw bytes of the image, or None if download fails.
        """
        # Check cache first
        cached = image_cache.get(url)
        if cached is not None:
            logger.info(f"Cache hit for image {url}")
            return cached

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to download image from {url}: HTTP {response.status}"
                        )
                        return None

                    content_type = response.headers.get("Content-Type", "")
                    # Only download if it looks like an image
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"URL {url} does not appear to be an image (Content-Type: {content_type})"
                        )
                        return None

                    data = await response.read()
                    if len(data) > max_size:
                        logger.warning(
                            f"Image from {url} exceeds max size ({len(data)} > {max_size} bytes)"
                        )
                        return None

                    logger.info(
                        f"Downloaded image from {url}: {len(data)} bytes, type: {content_type}"
                    )

                    # Store in cache
                    image_cache.set(url, data, content_type)
                    return data
        except aiohttp.ClientError as e:
            logger.warning(f"Error downloading image from {url}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading image from {url}")
            return None

    @staticmethod
    def build_image_content(base64_url: str) -> dict:
        """Build an image_url content part dict for the messages array.

        Args:
            base64_url: The base64-encoded data URL of the image.

        Returns:
            A dict like {"type": "image_url", "image_url": {"url": "..."}}
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": base64_url,
            },
        }

    @classmethod
    async def process_message_for_images(
        cls,
        message_content: str,
        max_size: int = MAX_IMAGE_DOWNLOAD_SIZE,
        max_pixels: int = 16_777_216,
    ) -> Tuple[List[dict], List[str]]:
        """Detect image URLs in a message, download and process them.

        Uses the unified image processing pipeline: each downloaded image
        is checked against the double-keyed ``processed_image_cache`` before
        being resized and compressed to WebP.

        Args:
            message_content: The text content of the message to scan.
            max_size: Maximum allowed download size in bytes per image.
            max_pixels: Maximum pixel count for resize (defaults to 4096×4096).

        Returns:
            Tuple of (list of image_url content parts, list of filenames/urls).
        """
        image_urls = cls.detect_image_urls(message_content)
        if not image_urls:
            return [], []

        content_parts: List[dict] = []
        sources: List[str] = []

        for url in image_urls:
            logger.info(f"Processing image URL in message: {url}")
            image_data = await cls.fetch_image(url, max_size)
            if image_data is None:
                logger.warning(f"Skipping image {url} — download failed")
                continue

            # Check the double-keyed processed cache
            pixel_hash = compute_pixel_hash(image_data)
            cached_processed = processed_image_cache.get(image_data, pixel_hash, max_pixels)

            if cached_processed is not None:
                data_url = build_webp_data_url(cached_processed)
                logger.info(
                    f"[ImageParsing] Processed cache hit for {url}: "
                    f"{len(cached_processed)} bytes"
                )
            else:
                processed = process_image_for_llm(image_data, max_pixels)
                if processed is None:
                    logger.warning(f"Skipping image {url} — processing failed")
                    continue
                processed_image_cache.set(image_data, pixel_hash, max_pixels, processed)
                data_url = build_webp_data_url(processed)
                logger.info(
                    f"[ImageParsing] Processed image from {url}: "
                    f"{len(image_data)} bytes → {len(processed)} bytes WebP"
                )

            content_part = cls.build_image_content(data_url)
            content_parts.append(content_part)
            sources.append(url)

        return content_parts, sources

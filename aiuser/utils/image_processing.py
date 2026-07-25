"""Shared image processing utilities for the aiuser cog.

Provides a unified pipeline for resizing and compressing images before
they are sent to the LLM.  All image sources (Discord attachments,
URL-fetched images, and generated images) go through the same pipeline
so the LLM always receives a consistent, size-limited WebP data URL.

The double-keyed ``processed_image_cache`` (in :mod:`image_cache`)
avoids redundant processing: images with identical content (by byte hash
or by pixel hash) hit the cache regardless of how they arrived.
"""

import base64
import hashlib
import logging
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("red.bz_cogs.aiuser")

# Default WebP quality for LLM-bound images (1-100).
DEFAULT_WEBP_QUALITY = 90


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_byte_hash(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data* (raw bytes)."""
    return hashlib.sha256(data).hexdigest()


def compute_pixel_hash(image_bytes: bytes) -> Optional[str]:
    """Return the hex SHA-256 of the canonical RGBA pixel data of an image.

    Decodes *image_bytes* with PIL, converts to RGBA, and hashes the raw
    pixel bytes.  This catches visually-identical images that have different
    file-level encodings (e.g. PNG metadata, JPEG quality, timestamps).

    Returns ``None`` if PIL cannot decode the image.
    """
    try:
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.load()
        canonical = pil_img.convert("RGBA").tobytes()
        return hashlib.sha256(canonical).hexdigest()
    except Exception as e:
        logger.debug(f"[ImageProcessing] Could not compute pixel hash: {e}")
        return None


# ---------------------------------------------------------------------------
# Resize + compress pipeline
# ---------------------------------------------------------------------------

def _scale_image(cv_image: np.ndarray, max_pixel_count: int) -> np.ndarray:
    """Scale *cv_image* down so its pixel count does not exceed *max_pixel_count*.

    Preserves the aspect ratio.  Returns the original image unchanged if it
    is already within the limit.  Uses ``INTER_AREA`` interpolation for
    high-quality down-scaling.
    """
    h, w = cv_image.shape[:2]
    if w * h <= max_pixel_count:
        return cv_image

    scale_ratio = np.sqrt(max_pixel_count / (w * h))
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    return cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_image_for_llm(
    original_bytes: bytes,
    max_pixels: int = 16_777_216,
    webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> Optional[bytes]:
    """Decode, resize, and compress *original_bytes* for sending to the LLM.

    Parameters
    ----------
    original_bytes:
        Raw image bytes in any format PIL/OpenCV can decode.
    max_pixels:
        Maximum total pixel count (width × height).  Images larger than this
        are scaled down preserving aspect ratio.  Defaults to 4096×4096.
    webp_quality:
        WebP compression quality (1-100).  Defaults to 90.

    Returns
    -------
    bytes or None
        The processed WebP image bytes, or ``None`` if decoding failed.
    """
    # Decode with OpenCV (ignores alpha, same as the existing pipeline).
    file_bytes_arr = np.frombuffer(original_bytes, dtype=np.uint8)
    cv_image = cv2.imdecode(file_bytes_arr, cv2.IMREAD_COLOR)
    if cv_image is None:
        logger.warning("[ImageProcessing] Failed to decode image with OpenCV")
        return None

    scaled = _scale_image(cv_image, max_pixels)

    params = [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
    success, encoded = cv2.imencode(".webp", scaled, params)
    if not success:
        logger.warning("[ImageProcessing] Failed to encode image to WebP")
        return None

    return encoded.tobytes()


def build_webp_data_url(processed_bytes: bytes) -> str:
    """Base64-encode *processed_bytes* and wrap in a ``data:image/webp`` URL."""
    b64 = base64.b64encode(processed_bytes).decode("utf-8")
    return f"data:image/webp;base64,{b64}"


# ---------------------------------------------------------------------------
# Token estimation (Gemini image tiling)
# ---------------------------------------------------------------------------

#: Flat cost for small images (both dimensions <= 384 px) and per-tile cost.
GEMINI_TOKENS_PER_TILE = 258
#: Images whose dimensions are both <= this are charged a flat tile.
GEMINI_SMALL_IMAGE_DIM = 384
#: Fallback estimate when dimensions cannot be determined.
FALLBACK_IMAGE_TOKENS = 756


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Gemini token cost for an image of *width* x *height* pixels.

    Gemini charges 258 tokens if both dimensions are <= 384 px.  Larger
    images are tiled into 768x768 tiles, each costing 258 tokens.  The
    number of tiles is approximated by a crop unit of
    ``floor(min(width, height) / 1.5)``; each dimension is divided by the
    crop unit and the results multiplied.  E.g. 960x540 -> crop unit 360
    -> ceil(960/360) * ceil(540/360) = 3 * 2 = 6 tiles -> 1548 tokens.
    """
    if width <= 0 or height <= 0:
        return FALLBACK_IMAGE_TOKENS
    if width <= GEMINI_SMALL_IMAGE_DIM and height <= GEMINI_SMALL_IMAGE_DIM:
        return GEMINI_TOKENS_PER_TILE
    crop_unit = max(1, int(min(width, height) // 1.5))
    tiles_w = -(-width // crop_unit)  # ceil division
    tiles_h = -(-height // crop_unit)
    return tiles_w * tiles_h * GEMINI_TOKENS_PER_TILE


def image_dimensions_from_data_url(data_url: str) -> Optional[tuple]:
    """Decode ``(width, height)`` from a base64 ``data:`` URL, or ``None``."""
    try:
        header, _, b64 = data_url.partition(",")
        if ";base64" not in header or not b64:
            return None
        raw = base64.b64decode(b64)
        with Image.open(BytesIO(raw)) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def estimate_image_part_tokens(image_url: str) -> int:
    """Estimate the token cost of an ``image_url`` content part.

    Decodes dimensions from a data URL when possible and applies the
    Gemini tiling formula; falls back to ``FALLBACK_IMAGE_TOKENS``.
    """
    dims = image_dimensions_from_data_url(image_url)
    if dims is None:
        return FALLBACK_IMAGE_TOKENS
    return estimate_image_tokens(dims[0], dims[1])


def estimate_file_part_tokens(file_data: str) -> int:
    """Rough token estimate for a base64 ``file`` content part.

    Base64 expands raw bytes by 4/3 and text tokenizes at ~4 chars/token,
    so tokens ~= b64_length * 3/16.
    """
    if not file_data:
        return 0
    b64_len = len(file_data.partition(",")[2])
    return max(1, (b64_len * 3) // 16)


def process_image_for_llm_content(
    original_bytes: bytes,
    max_pixels: int = 16_777_216,
    webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> Optional[dict]:
    """Convenience: process and return an OpenAI-compatible image_url content part.

    Returns a dict like::

        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,..."}}

    or ``None`` if processing failed.
    """
    processed = process_image_for_llm(original_bytes, max_pixels, webp_quality)
    if processed is None:
        return None
    return {
        "type": "image_url",
        "image_url": {"url": build_webp_data_url(processed)},
    }

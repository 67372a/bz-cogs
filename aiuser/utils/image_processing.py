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

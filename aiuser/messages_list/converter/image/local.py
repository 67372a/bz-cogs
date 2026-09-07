import logging
from typing import Optional

import pytesseract
from discord import Message
from PIL import Image
from redbot.core.data_manager import cog_data_path
from transformers import BlipForConditionalGeneration, BlipProcessor

from aiuser.types.abc import MixinMeta
from aiuser.common.utilities import to_thread
from aiuser.utils.image_cache import caption_cache

logger = logging.getLogger("red.bz_cogs.aiuser")


async def process_image_locally(
    cog: MixinMeta,
    message: Message,
    image: Image.Image,
    pixel_hash: Optional[str] = None,
    max_pixels: Optional[int] = None,
):
    """Caption/transcribe *image* locally.

    When ``pixel_hash``/``max_pixels`` are provided, the BLIP caption is
    pinned in the caption cache so the serialized context content for this
    image is byte-stable across requests (prompt-cache friendly) and the
    model is not re-run on every conversion.
    """
    path = cog_data_path(cog)
    scanned_text = await extract_text(image)
    author = message.author

    if scanned_text and len(scanned_text.split()) > 10:
        content = f'User "{author.name}" with display name "{author.display_name}" sent: [Image saying "{scanned_text}"]'
    else:
        caption = None
        if pixel_hash is not None and max_pixels is not None:
            caption = caption_cache.get(pixel_hash, max_pixels)
        if caption is None:
            caption = await caption_image(image, path)
            if pixel_hash is not None and max_pixels is not None:
                caption_cache.set(pixel_hash, max_pixels, caption)
        content = f'User "{author.name}" with display name "{author.display_name}" sent: [Image: {caption}]'
    return content


@to_thread()
def caption_image(image: Image.Image, datapath):
    cache_path = datapath or "~/.cache/huggingface/datasets"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_path)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", cache_dir=cache_path)

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)

    caption = (processor.decode(out[0], skip_special_tokens=True))

    return caption


@to_thread()
def extract_text(image: Image.Image):
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, timeout=30)
    text = " ".join(word for i, word in enumerate(data["text"])
                    if int(data["conf"][i]) >= 60)
    return text.strip()

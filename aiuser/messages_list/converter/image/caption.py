import base64
import logging
from io import BytesIO

from discord import Message
from PIL import Image
import cv2
import numpy as np

# ... (rest of your imports)
from aiuser.types.abc import MixinMeta
from aiuser.types.enums import ScanImageMode
from aiuser.messages_list.converter.helpers import format_text_content
from aiuser.messages_list.converter.image.AI_horde import \
    process_image_ai_horde

logger = logging.getLogger("red.bz_cogs.aiuser")


async def transcribe_image(cog: MixinMeta, message: Message):
    config = cog.config
    attachment = message.attachments[0]
    mode = ScanImageMode(await config.guild(message.guild).scan_images_mode())

    buffer = BytesIO()
    await attachment.save(buffer)

    # The buffer is converted to a NumPy array, then decoded by OpenCV
    file_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    # cv2.IMREAD_COLOR ignores transparency, use cv2.IMREAD_UNCHANGED to keep it
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_image is None:
        logger.error(f"Failed to decode image from attachment in message {message.id}")
        return None

    maxsize = 4096*4096 if mode == ScanImageMode.LLM else 1024*1024
    scaled_cv_image = scale_image(cv_image, maxsize)

    content = await process_image(cog, message, scaled_cv_image, mode)

    if content and mode != ScanImageMode.LLM:
        cog.cached_messages[message.id] = content

    return content


async def process_image(cog: MixinMeta, message: Message, cv_image: np.ndarray, mode: ScanImageMode):
    if mode == ScanImageMode.AI_HORDE:
        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        return await process_image_ai_horde(cog, message, image)
    elif mode == ScanImageMode.LOCAL:
        try:
            from aiuser.messages_list.converter.image.local import \
                process_image_locally
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return await process_image_locally(cog, message, image)
        except ImportError:
            logger.exception("Local image scanning dependencies not installed, check cog README for instructions")
            return None
    elif mode == ScanImageMode.LLM:
        content = []
        if message.content != "":
            content.append({"type": "text", "text": format_text_content(message)})

        # 1. Define the WebP quality. 1-100. Higher is better quality and larger size.
        #    85-95 is a great range for a good balance. 90 is a solid default.
        webp_quality = 90

        # 2. Encode the OpenCV image to WebP format in memory
        #    Note the '.webp' extension and the quality parameter.
        params = [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
        success, encoded_image = cv2.imencode('.webp', cv_image, params)

        if not success:
            logger.error("Failed to encode image to WebP format")
            return None

        # Base64 encode the bytes of the PNG
        base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        content.append(
            {"type": "image_url", "image_url": {
             "url": f"data:image/webp;base64,{base64_image}"}
             })
        return content
    else:
        return None


def scale_image(cv_image: np.ndarray, max_pixel_count: int) -> np.ndarray:
    """
    Scales down an OpenCV image (NumPy array) to not exceed a maximum
    pixel count, preserving aspect ratio.

    :param cv_image: The input OpenCV image (as a NumPy array).
    :param max_pixel_count: The maximum total number of pixels (width * height).
    :return: The scaled OpenCV image (as a NumPy array).
    """
    original_height, original_width = cv_image.shape[:2]
    original_pixel_count = original_width * original_height

    if original_pixel_count <= max_pixel_count:
        logger.debug("Image is already smaller than the max pixel count. No scaling needed.")
        return cv_image

    # Calculate scaling ratio
    scale_ratio = np.sqrt(max_pixel_count / original_pixel_count)
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    logger.info(f"Scaling image from {original_width}x{original_height} to {new_width}x{new_height}")

    # Resize using INTER_AREA for high-quality shrinking
    scaled_image = cv2.resize(
        cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return scaled_image
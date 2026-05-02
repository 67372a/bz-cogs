import asyncio
import base64
import logging
import re
from typing import Dict, List, Optional, Tuple

import aiohttp
from redbot.core import Config, commands

from aiuser.types.enums import OpenRouterToolType
from aiuser.types.openrouter_types import (
    PdfParsingParameters,
    deserialize_parameters,
)

logger = logging.getLogger("red.bz_cogs.aiuser")

# Matches URLs that end in .pdf, optionally followed by query params
PDF_URL_PATTERN = re.compile(
    r"(https?://[^\s<>\"']+?\.pdf)(\?[^\s<>\"']*)?(?=[\s<>\"']|$)",
    re.IGNORECASE,
)

# Matches generic URLs that might be PDFs (content-disposition headers)
GENERIC_URL_PATTERN = re.compile(r"(https?://[^\s<>\"']+)", re.IGNORECASE)

# Standard Discord CDN URL pattern (these are native Discord attachments)
DISCORD_CDN_URL_PATTERN = re.compile(
    r"https?://cdn\.discordapp\.com/attachments/[^\s<>\"']+",
    re.IGNORECASE,
)

# Maximum PDF file size for download (10 MB default)
MAX_PDF_DOWNLOAD_SIZE = 10 * 1024 * 1024


class OpenRouterPdfParsing:
    """Handles PDF detection, retrieval, and encoding for OpenRouter server-side parsing.

    This is NOT a ToolCall subclass. It is a pre-processor service that:
    1. Detects PDF links in Discord message text
    2. Downloads PDFs from URLs and base64-encodes them
    3. Builds file content parts for the messages array
    4. Manages annotation caching for reply chains
    """

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx

    @classmethod
    def detect_pdf_links(cls, message_content: str) -> List[str]:
        """Scan message text for PDF URLs.

        Args:
            message_content: The raw text content of a Discord message.

        Returns:
            List of unique PDF URLs found in the text.
        """
        if not message_content:
            return []

        matches = PDF_URL_PATTERN.findall(message_content)
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
    async def fetch_pdf(cls, url: str) -> Optional[bytes]:
        """Download a PDF from a URL.

        Args:
            url: The URL of the PDF to download.

        Returns:
            Raw bytes of the PDF, or None if download fails.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, max_size=MAX_PDF_DOWNLOAD_SIZE) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to download PDF from {url}: HTTP {response.status}"
                        )
                        return None

                    content_type = response.headers.get("Content-Type", "")
                    # Only download if it looks like a PDF
                    if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
                        logger.warning(
                            f"URL {url} does not appear to be a PDF (Content-Type: {content_type})"
                        )
                        return None

                    data = await response.read()
                    logger.info(
                        f"Downloaded PDF from {url}: {len(data)} bytes"
                    )
                    return data
        except aiohttp.ClientError as e:
            logger.warning(f"Error downloading PDF from {url}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading PDF from {url}")
            return None

    @staticmethod
    def encode_pdf_to_base64(pdf_data: bytes) -> str:
        """Encode raw PDF bytes to a base64 data URL.

        Args:
            pdf_data: Raw bytes of the PDF file.

        Returns:
            A data URL string like "data:application/pdf;base64,..."
        """
        encoded = base64.b64encode(pdf_data).decode("utf-8")
        return f"data:application/pdf;base64,{encoded}"

    @staticmethod
    def extract_filename_from_url(url: str, index: int = 0) -> str:
        """Extract a filename from a PDF URL.

        Args:
            url: The PDF URL.
            index: Index for generating fallback filenames.

        Returns:
            A filename string.
        """
        # Try to extract filename from URL path
        path = url.split("?")[0]
        filename = path.rsplit("/", 1)[-1]
        if filename.lower().endswith(".pdf"):
            return filename
        return f"document_{index + 1}.pdf"

    @staticmethod
    def build_file_content(filename: str, file_data: str) -> dict:
        """Build a file content part dict for the messages array.

        Args:
            filename: The display filename for the PDF.
            file_data: The base64-encoded data URL or direct URL.

        Returns:
            A dict like {"type": "file", "file": {"filename": "...", "file_data": "..."}}
        """
        return {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": file_data,
            },
        }

    async def build_plugins(self) -> List[dict]:
        """Build the plugins array with configured PDF parsing engine.

        Returns:
            A list containing the file-parser plugin dict.
        """
        params_json = await self.config.guild(self.ctx.guild).openrouter_pdf_parsing_parameters()
        params: PdfParsingParameters = deserialize_parameters(
            params_json, OpenRouterToolType.PDF_PARSING
        )
        engine = params.engine or "cloudflare-ai"
        return [{"id": "file-parser", "pdf": {"engine": engine}}]

    async def process_message_for_pdfs(
        self, message_content: str
    ) -> Tuple[List[dict], List[str]]:
        """Detect PDF URLs in a message, download and encode them.

        Args:
            message_content: The text content of the message to scan.

        Returns:
            Tuple of (list of file content parts, list of filenames).
        """
        pdf_urls = self.detect_pdf_links(message_content)
        if not pdf_urls:
            return [], []

        file_contents: List[dict] = []
        filenames: List[str] = []

        for i, url in enumerate(pdf_urls):
            logger.info(f"Processing PDF URL in message: {url}")
            pdf_data = await self.fetch_pdf(url)
            if pdf_data is None:
                logger.warning(f"Skipping PDF {url} — download failed")
                continue

            encoded = self.encode_pdf_to_base64(pdf_data)
            filename = self.extract_filename_from_url(url, i)
            file_content = self.build_file_content(filename, encoded)
            file_contents.append(file_content)
            filenames.append(filename)

            logger.info(f"Encoded PDF {filename}: {len(pdf_data)} bytes -> {len(encoded)} char data URL")

        return file_contents, filenames
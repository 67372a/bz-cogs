import logging
import os
from typing import Dict, List

from redbot.core import Config, commands

from aiuser.functions.tool_call import ToolCall
from aiuser.functions.types import Function, Parameters, ToolCallSchema

logger = logging.getLogger("red.bz_cogs.aiuser")

# Discord file size limits
MAX_SINGLE_FILE_SIZE = 25 * 1024 * 1024  # 25 MB per file
MAX_TOTAL_FILES = 10  # Maximum number of files per call


attach_files_tool_call_schema = ToolCallSchema(
    function=Function(
        name="attach_files",
        description=(
            "Attaches one or more code or text files to your response as Discord file "
            "attachments. Use this instead of pasting code directly in messages when the "
            "code is too long, spans multiple files, or would break formatting in a "
            "Discord message. The files will be sent as downloadable attachments alongside "
            "your text response."
        ),
        parameters=Parameters(
            properties={
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": (
                                    "The desired filename including the file extension "
                                    "(e.g. 'main.py', 'style.css', 'config.json', 'README.md')."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "The full text content to place into the file."
                                ),
                            },
                        },
                        "required": ["filename", "content"],
                    },
                    "description": (
                        "Array of file objects to attach. Each object must have a "
                        "'filename' (including extension) and 'content' (the full text data)."
                    ),
                },
            },
            required=["files"],
        ),
    )
)


class AttachFilesToolCall(ToolCall):
    """Attach code or text files as Discord message attachments.

    This tool allows the LLM to output code and text files as downloadable
    Discord attachments rather than pasting code inline in messages. This
    keeps code organized, correctly formatted, and easy for users to save.

    Attached files are stored internally and collected by the pipeline after
    execution, flowing into the same response path as generated images
    (``collected_images`` -> ``PipelineResult.images`` -> Discord file
    attachments).
    """

    schema = attach_files_tool_call_schema
    function_name = attach_files_tool_call_schema.function.name

    def __init__(self, config: Config, ctx: commands.Context):
        super().__init__(config, ctx)
        self.attached_files: List[Dict] = []

    def get_attached_files(self) -> List[Dict]:
        """Return stored attached file data and clear the internal list.

        Each entry has:
            - bytes: UTF-8 encoded file content.
            - filename: The filename (including extension) for the Discord attachment.

        Returns:
            List of file dicts ready to be sent as Discord attachments.
        """
        files = self.attached_files[:]
        self.attached_files.clear()
        return files

    async def _handle(self, arguments: dict) -> str:
        files_input = arguments.get("files", [])

        if not files_input:
            return "Error: No files were provided. Supply at least one file with 'filename' and 'content'."

        if len(files_input) > MAX_TOTAL_FILES:
            return f"Error: Too many files provided. Maximum allowed is {MAX_TOTAL_FILES}, got {len(files_input)}."

        attached_names: List[str] = []
        errors: List[str] = []

        for i, file_data in enumerate(files_input):
            filename = file_data.get("filename", "").strip()
            content = file_data.get("content")

            if not filename:
                errors.append(f"File {i + 1}: Empty filename.")
                continue

            if content is None:
                errors.append(f"File {i + 1} ('{filename}'): Empty content.")
                continue

            # Sanitize filename: strip directory components, keep only basename
            filename = os.path.basename(filename)
            if not filename or filename.startswith("."):
                errors.append(f"File {i + 1}: Invalid filename after sanitization.")
                continue

            # Encode content to bytes
            try:
                file_bytes = content.encode("utf-8")
            except UnicodeEncodeError as e:
                errors.append(f"File {i + 1} ('{filename}'): Encoding error — {e}")
                continue

            # Check file size
            if len(file_bytes) > MAX_SINGLE_FILE_SIZE:
                size_mb = len(file_bytes) / (1024 * 1024)
                errors.append(
                    f"File '{filename}': Size ({size_mb:.1f} MB) exceeds the "
                    f"{MAX_SINGLE_FILE_SIZE // (1024 * 1024)} MB limit."
                )
                continue

            self.attached_files.append({
                "bytes": file_bytes,
                "filename": filename,
            })
            attached_names.append(filename)

        # Build response message
        parts = []
        if attached_names:
            parts.append(
                f"Successfully attached {len(attached_names)} file(s): "
                f"{', '.join(f'`{n}`' for n in attached_names)}."
            )
        if errors:
            parts.append("Errors:\n" + "\n".join(errors))

        if not attached_names:
            return "Error: No files could be attached.\n" + "\n".join(errors)

        return "\n".join(parts)

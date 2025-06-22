
from redbot.core import Config, commands
from redbot.core.bot import Red

from aiuser.functions.types import ToolCallSchema


class ToolCall:
    schema: ToolCallSchema = None
    function_name: str = None

    def __init__(self, config: Config, ctx: commands.Context):
        self.config = config
        self.ctx = ctx
        self.bot: Red = ctx.bot

    def run(self, arguments: dict):
        return self._handle(arguments)

    def _handle(arguments: dict):
        raise NotImplementedError


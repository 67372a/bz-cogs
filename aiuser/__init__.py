import aiohttp

# Compatibility shims for openai 3.x vendored httpx_aiohttp transport
if not hasattr(aiohttp, "SocketTimeoutError"):
    aiohttp.SocketTimeoutError = getattr(aiohttp, "ServerTimeoutError", TimeoutError)

if not hasattr(aiohttp.client_exceptions, "NonHttpUrlClientError"):
    aiohttp.client_exceptions.NonHttpUrlClientError = getattr(
        aiohttp.client_exceptions,
        "NonHttpUrlRedirectClientError",
        getattr(aiohttp.client_exceptions, "InvalidURL", aiohttp.ClientError),
    )

from .core.aiuser import AIUser
from redbot.core.utils import get_end_user_data_statement

__red_end_user_data_statement__ = get_end_user_data_statement(__file__)

async def setup(bot):
    await bot.add_cog(AIUser(bot))

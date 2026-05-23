"""Helper to import aiuser.functions.openrouter.* modules directly.

Uses the same technique as mock_importer.py — bypasses __init__.py side effects
by pre-registering dummy packages in sys.modules. Also mocks the sub-dependencies
that these function modules require (redbot.core.Config, etc.).
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

from tests.mock_importer import _make_mock_package, import_module_directly

# ---------------------------------------------------------------------------
# Ensure the aiuser and redbot package trees are mocked
# ---------------------------------------------------------------------------
_packages = [
    "aiuser",
    "aiuser.types",
    "aiuser.functions",
    "aiuser.functions.openrouter",
    "aiuser.utils",
]
for pkg in _packages:
    if pkg not in sys.modules:
        sys.modules[pkg] = _make_mock_package(pkg)

# ---------------------------------------------------------------------------
# Mock redbot.core with real module objects so attr access works
# ---------------------------------------------------------------------------
import types as _types

def _make_module(name, attrs=None):
    mod = _types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    mod.__file__ = f"<mocked-{name}>"
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

_redbot = _make_module("redbot", {
    "__version__": "3.5.0",
    "version_info": MagicMock(),
})
sys.modules["redbot"] = _redbot

_redbot_core = _make_module("redbot.core", {
    "Config": MagicMock(),
    "commands": MagicMock(),
    "app_commands": MagicMock(),
})
sys.modules["redbot.core"] = _redbot_core

_redbot_sub = {
    "redbot.core.commands": MagicMock(),
    "redbot.core.bot": MagicMock(),
    "redbot.core.utils": MagicMock(),
    "redbot.core.utils.chat_formatting": MagicMock(),
    "redbot.core.utils.views": MagicMock(),
}
for name, mod in _redbot_sub.items():
    sys.modules[name] = mod

# ---------------------------------------------------------------------------
# Pre-register aiuser sub-modules that the function classes import
# ---------------------------------------------------------------------------
_aiuser_preload = {
    "aiuser.types.enums": MagicMock(),
    "aiuser.types.openrouter_types": MagicMock(),
}
for name, mod in _aiuser_preload.items():
    sys.modules[name] = mod

# ---------------------------------------------------------------------------
# Actually load the real enums and openrouter_types
# ---------------------------------------------------------------------------
sys.modules["aiuser.types.enums"] = import_module_directly(
    "aiuser.types.enums", "aiuser/types/enums.py"
)
sys.modules["aiuser.types.openrouter_types"] = import_module_directly(
    "aiuser.types.openrouter_types", "aiuser/types/openrouter_types.py"
)

# ---------------------------------------------------------------------------
# Now load the three function modules
# ---------------------------------------------------------------------------
sys.modules["aiuser.functions.openrouter.web_search"] = import_module_directly(
    "aiuser.functions.openrouter.web_search", "aiuser/functions/openrouter/web_search.py"
)
sys.modules["aiuser.functions.openrouter.web_fetch"] = import_module_directly(
    "aiuser.functions.openrouter.web_fetch", "aiuser/functions/openrouter/web_fetch.py"
)
sys.modules["aiuser.functions.openrouter.image_generation"] = import_module_directly(
    "aiuser.functions.openrouter.image_generation", "aiuser/functions/openrouter/image_generation.py"
)
sys.modules["aiuser.functions.openrouter.pdf_parsing"] = import_module_directly(
    "aiuser.functions.openrouter.pdf_parsing", "aiuser/functions/openrouter/pdf_parsing.py"
)
sys.modules["aiuser.functions.openrouter.image_parsing"] = import_module_directly(
    "aiuser.functions.openrouter.image_parsing", "aiuser/functions/openrouter/image_parsing.py"
)

# Load the real image_cache module (it has no heavy deps)
sys.modules["aiuser.utils.image_cache"] = import_module_directly(
    "aiuser.utils.image_cache", "aiuser/utils/image_cache.py"
)

# Re-export
OpenRouterWebSearch = sys.modules["aiuser.functions.openrouter.web_search"].OpenRouterWebSearch
OpenRouterWebFetch = sys.modules["aiuser.functions.openrouter.web_fetch"].OpenRouterWebFetch
OpenRouterImageGeneration = sys.modules["aiuser.functions.openrouter.image_generation"].OpenRouterImageGeneration
OpenRouterPdfParsing = sys.modules["aiuser.functions.openrouter.pdf_parsing"].OpenRouterPdfParsing
OpenRouterImageParsing = sys.modules["aiuser.functions.openrouter.image_parsing"].OpenRouterImageParsing

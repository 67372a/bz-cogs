"""Helper to import aiuser modules directly without triggering aiuser/__init__.py."""

import importlib.util
import sys
from types import ModuleType
from typing import Optional


def _make_mock_package(name: str) -> ModuleType:
    """Create a dummy package entry in sys.modules to prevent real init from running."""
    pkg = ModuleType(name)
    pkg.__path__ = []
    pkg.__package__ = name
    pkg.__file__ = f"<mocked-{name}>"
    return pkg


def import_module_directly(qualified_name: str, filepath: str) -> ModuleType:
    """Import a single module by file path, bypassing any __init__ side effects.

    Registers the module in sys.modules so further ``from ... import``
    statements within *other* modules will resolve to this same object.

    Args:
        qualified_name:  e.g. ``"aiuser.types.enums"``
        filepath:        path relative to cwd, e.g. ``"aiuser/types/enums.py"``

    Returns:
        The loaded module object.
    """
    # Ensure the parent package(s) exist as mock packages so Python's import
    # machinery doesn't try to run the real __init__.py.
    parts = qualified_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = _make_mock_package(parent)

    spec = importlib.util.spec_from_file_location(qualified_name, filepath)
    if spec is None:
        raise ImportError(f"Could not find spec for {qualified_name} @ {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_mock_package(name: str):
    """Ensure a dummy package entry exists in sys.modules for *name*."""
    if name not in sys.modules:
        sys.modules[name] = _make_mock_package(name)

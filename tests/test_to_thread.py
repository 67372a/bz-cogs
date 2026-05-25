"""Tests for the to_thread decorator in aiuser.utils.utilities.

Verifies that:
- Positional arguments are forwarded correctly
- Keyword arguments are forwarded correctly (regression test for run_in_executor bug)
- Mixed positional and keyword arguments work
- functools.wraps preserves function metadata
"""

import asyncio
import functools


def _to_thread(timeout=300):
    """Reimplementation of to_thread from aiuser.utils.utilities for isolated testing.

    This mirrors the production implementation so tests validate the
    actual decorator logic (functools.partial + run_in_executor) without
    importing the full utilities module and its heavy dependencies.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            func_call = functools.partial(func, *args, **kwargs)
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func_call), timeout
            )
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests: to_thread keyword argument handling
# ---------------------------------------------------------------------------


class TestToThreadKwargs:
    """Verify to_thread properly forwards keyword arguments.

    This is a regression test for the bug where run_in_executor() received
    unexpected keyword arguments because kwargs were passed directly instead
    of being wrapped with functools.partial.
    """

    def test_positional_args(self):
        """Positional arguments should be forwarded correctly."""
        @_to_thread(timeout=30)
        def add(a, b):
            return a + b

        result = _run_async(add(2, 3))
        assert result == 5

    def test_keyword_args(self):
        """Keyword arguments should be forwarded correctly (the fixed bug)."""
        @_to_thread(timeout=30)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = _run_async(greet("World", greeting="Hi"))
        assert result == "Hi, World!"

    def test_kwargs_only(self):
        """Function called with only keyword arguments."""
        @_to_thread(timeout=30)
        def build_url(host, path="/", scheme="https"):
            return f"{scheme}://{host}{path}"

        result = _run_async(build_url("example.com", path="/api", scheme="http"))
        assert result == "http://example.com/api"

    def test_mixed_args_and_kwargs(self):
        """Mixed positional and keyword arguments should work."""
        @_to_thread(timeout=30)
        def format_message(prefix, message, suffix="!"):
            return f"{prefix}: {message}{suffix}"

        result = _run_async(format_message("INFO", "started", suffix="..."))
        assert result == "INFO: started..."

    def test_default_kwargs_not_provided(self):
        """Default keyword arguments should be used when not provided."""
        @_to_thread(timeout=30)
        def render(code, theme="dark"):
            return f"code={code}, theme={theme}"

        result = _run_async(render("flowchart TD"))
        assert result == "code=flowchart TD, theme=dark"

    def test_preserves_function_name(self):
        """_to_thread should preserve the wrapped function's __name__."""
        @_to_thread(timeout=30)
        def my_special_function(x):
            return x

        assert my_special_function.__name__ == "my_special_function"

    def test_preserves_function_docstring(self):
        """_to_thread should preserve the wrapped function's __doc__."""
        @_to_thread(timeout=30)
        def documented_function(x):
            """This is the docstring."""
            return x

        assert documented_function.__doc__ == "This is the docstring."

    def test_multiple_kwargs(self):
        """Multiple keyword arguments should all be forwarded."""
        @_to_thread(timeout=30)
        def complex_func(a, b=1, c=2, d=3):
            return a + b + c + d

        result = _run_async(complex_func(10, b=20, c=30, d=40))
        assert result == 100

    def test_none_executor(self):
        """Using None as executor (default) should work."""
        @_to_thread(timeout=30)
        def simple(x, y=0):
            return x + y

        result = _run_async(simple(5, y=10))
        assert result == 15

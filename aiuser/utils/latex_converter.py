"""Convert LaTeX math expressions in LLM responses to Discord-safe Unicode/plaintext.

This module identifies LaTeX math delimiters (``$...$``, ``$$...$$``, ``\\(...\\)``,
``\\[...]``) in mixed markdown/text, converts the LaTeX content to Unicode symbols
or readable plaintext, and returns the result safe for Discord embed rendering.

Key safety features:
- Code blocks (fenced and inline) are extracted and preserved before any processing.
- Heuristic filtering avoids false positives on currency ($100) and shell vars ($HOME).
- Unbalanced delimiters are left untouched.
- Complex expressions that cannot be cleanly converted fall back to markdown code blocks.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger("red.bz_cogs.aiuser")


# ---------------------------------------------------------------------------
#  Unicode mapping tables
# ---------------------------------------------------------------------------

# Greek letters (lowercase)
GREEK_LOWER = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\varepsilon": "ε", r"\vartheta": "ϑ", r"\varphi": "φ",
    r"\varpi": "ϖ", r"\varrho": "ϱ", r"\varsigma": "ς",
}

# Greek letters (uppercase)
GREEK_UPPER = {
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

# Binary operators
OPERATORS = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±",
    r"\mp": "∓", r"\ast": "∗", r"\star": "⋆", r"\circ": "∘",
    r"\bullet": "•", r"\oplus": "⊕", r"\otimes": "⊗",
    r"\odot": "⊙", r"\dagger": "†", r"\ddagger": "‡",
    r"\cap": "∩", r"\cup": "∪", r"\sqcap": "⊓", r"\sqcup": "⊔",
    r"\vee": "∨", r"\wedge": "∧", r"\setminus": "∖",
    r"\wr": "≀", r"\diamond": "⋄", r"\bigtriangleup": "△",
    r"\bigtriangledown": "▽", r"\triangleleft": "◃", r"\triangleright": "▹",
    r"\lhd": "⊲", r"\rhd": "⊳", r"\unlhd": "⊴", r"\unrhd": "⊵",
    r"\bigcirc": "◯",
}

# Relations
RELATIONS = {
    r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\sim": "∼", r"\simeq": "≃", r"\cong": "≅", r"\propto": "∝",
    r"\lt": "<", r"\gt": ">", r"\ll": "≪", r"\gg": "≫",
    r"\prec": "≺", r"\succ": "≻", r"\preceq": "⪯", r"\succeq": "⪰",
    r"\subset": "⊂", r"\supset": "⊃", r"\subseteq": "⊆", r"\supseteq": "⊇",
    r"\sqsubset": "⊏", r"\sqsupset": "⊐", r"\sqsubseteq": "⊑", r"\sqsupseteq": "⊒",
    r"\in": "∈", r"\ni": "∋", r"\notin": "∉", r"\perp": "⊥",
    r"\mid": "∣", r"\parallel": "∥", r"\doteq": "≐",
    r"\models": "⊨", r"\bowtie": "⋈", r"\smile": "⌣", r"\frown": "⌢",
}

# Arrows
ARROWS = {
    r"\leftarrow": "←", r"\gets": "←", r"\rightarrow": "→", r"\to": "→",
    r"\leftrightarrow": "↔", r"\Leftarrow": "⇐", r"\Rightarrow": "⇒",
    r"\Leftrightarrow": "⇔", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\updownarrow": "↕", r"\Uparrow": "⇑", r"\Downarrow": "⇓",
    r"\Updownarrow": "⇕", r"\mapsto": "↦", r"\longleftarrow": "⟵",
    r"\longrightarrow": "⟶", r"\longleftrightarrow": "⟷",
    r"\Longleftarrow": "⟸", r"\Longrightarrow": "⟹",
    r"\Longleftrightarrow": "⟺", r"\nearrow": "↗", r"\searrow": "↘",
    r"\swarrow": "↙", r"\nwarrow": "↖",
    r"\hookleftarrow": "↩", r"\hookrightarrow": "↪",
    r"\rightleftharpoons": "⇌",
}

# Miscellaneous symbols
MISC_SYMBOLS = {
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄",
    r"\emptyset": "∅", r"\varnothing": "∅",
    r"\top": "⊤", r"\bot": "⊥",
    r"\angle": "∠", r"\measuredangle": "∡",
    r"\neg": "¬", r"\lnot": "¬",
    r"\flat": "♭", r"\natural": "♮", r"\sharp": "♯",
    r"\ell": "ℓ", r"\hbar": "ℏ", r"\Re": "ℜ", r"\Im": "ℑ",
    r"\aleph": "ℵ", r"\beth": "ℶ", r"\gimel": "ג",
    r"\wp": "℘", r"\complement": "∁",
    r"\prime": "′", r"\prime": "′",
    r"\ldots": "…", r"\cdots": "⋯", r"\vdots": "⋮", r"\ddots": "⋱",
    r"\dots": "…",
    r"\therefore": "∴", r"\because": "∵",
    r"\triangle": "△", r"\square": "□",
    r"\checkmark": "✓", r"\clubsuit": "♣", r"\diamondsuit": "♢",
    r"\heartsuit": "♡", r"\spadesuit": "♠",
    r"\dag": "†", r"\ddag": "‡", r"\S": "§", r"\P": "¶",
}

# Large operators (usually rendered differently in display mode)
LARGE_OPERATORS = {
    r"\sum": "Σ", r"\prod": "Π", r"\coprod": "∐",
    r"\int": "∫", r"\iint": "∬", r"\iiint": "∭", r"\oint": "∮",
    r"\bigcap": "⋂", r"\bigcup": "⋃", r"\bigsqcup": "⨆",
    r"\bigvee": "⋁", r"\bigwedge": "⋀",
    r"\bigotimes": "⨂", r"\bigoplus": "⨁", r"\bigodot": "⨀",
    r"\biguplus": "⨄",
}

# Delimiters
DELIMITERS = {
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
    r"\langle": "⟨", r"\rangle": "⟩",
}

# Superscript characters (Unicode)
_SUPERSCRIPT_MAP = str.maketrans(
    "0123456789+-=()niABDEGHIJKLMNOPRTUVWabcdefghjklmnoprstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵃᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
)

# Subscript characters (Unicode)
_SUBSCRIPT_MAP = str.maketrans(
    "0123456789+-=()aehijklmnoprstuvx",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
)

# Merge all simple command mappings
ALL_SIMPLE_MAPS = {
    **GREEK_LOWER, **GREEK_UPPER, **OPERATORS, **RELATIONS,
    **ARROWS, **MISC_SYMBOLS, **LARGE_OPERATORS, **DELIMITERS,
}

# Commands whose argument should be rendered in a specific style
FONT_COMMANDS = {
    r"\mathbf", r"\mathit", r"\mathsf", r"\mathrm",
    r"\textbf", r"\textit", r"\textrm", r"\text",
    r"\boldsymbol", r"\mathbb", r"\mathcal", r"\mathfrak",
    r"\operatorname",
}

# Pattern to match a LaTeX command name (backslash + letters)
_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")


# ---------------------------------------------------------------------------
#  Code block protection
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")


# Mutable counter for placeholder uniqueness across multiple _extract_blocks calls.
# Using a list so nested closures can mutate it without 'global'/'nonlocal' issues.
_BLOCK_COUNTER = [0]


def _extract_blocks(text: str, pattern: re.Pattern) -> Tuple[str, dict]:
    """Replace matches of *pattern* with placeholders, returning modified text and a restore map."""
    placeholders = {}

    def replacer(m):
        key = f"\x00BLOCK{_BLOCK_COUNTER[0]}\x00"
        _BLOCK_COUNTER[0] += 1
        placeholders[key] = m.group(0)
        return key

    new_text = pattern.sub(replacer, text)
    return new_text, placeholders


def _restore_blocks(text: str, placeholders: dict) -> str:
    """Restore all placeholders back to their original content."""
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


# ---------------------------------------------------------------------------
#  Delimiter detection
# ---------------------------------------------------------------------------

# Delimiter patterns (ordered by safety — process most-confident first)
_DISPLAY_DOUBLE_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_DISPLAY_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_PAREN = re.compile(r"\\\((.+?)\\\)")
_INLINE_SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Heuristic: looks like a currency amount ($100, $3.50, $1,234.56)
_CURRENCY_RE = re.compile(r"^\s*\d[\d,]*\.?\d*\s*$")

# Heuristic: content has at least one LaTeX-specific token
_HAS_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_HAS_LATEX_GROUP = re.compile(r"\{[^}]*\}")
_HAS_SUBSUPER = re.compile(r"[_^]\{")  # _{ or ^{
_HAS_SUBSUPER_SINGLE = re.compile(r"[_^][a-zA-Z0-9](?=[^a-zA-Z0-9]|$)")  # _x or ^2 at boundary


def _is_likely_latex(content: str) -> bool:
    """Heuristic check: does the content inside a single-$ delimiter look like LaTeX?

    Returns True if it contains LaTeX commands, grouped braces, or subscript/superscript
    syntax. Returns False for things like ``$100`` or ``$PATH``.
    """
    stripped = content.strip()

    # Currency: purely numeric with optional decimal/comma
    if _CURRENCY_RE.match(stripped):
        return False

    # Empty
    if not stripped:
        return False

    # Contains a backslash command
    if _HAS_LATEX_CMD.search(stripped):
        return True

    # Contains brace grouping with sub/super: _{...} or ^{...}
    if _HAS_SUBSUPER.search(stripped):
        return True

    # Contains a general brace group (could be LaTeX)
    if _HAS_LATEX_GROUP.search(stripped):
        return True

    # Single-char sub/super: x^2, a_b
    if _HAS_SUBSUPER_SINGLE.search(stripped):
        return True

    return False


def _find_balanced_delimiters(text: str, open_delim: str, close_delim: str) -> List[Tuple[int, int, str]]:
    """Find all balanced open/close delimiter pairs in text.

    Returns list of (start_index, end_index, content) tuples where
    start_index and end_index span the full delimiters (inclusive).
    """
    results = []
    i = 0
    open_len = len(open_delim)
    close_len = len(close_delim)

    while i < len(text):
        if text[i:i + open_len] == open_delim:
            # Found opening delimiter, now find matching close
            depth = 1
            j = i + open_len
            while j < len(text) and depth > 0:
                if text[j:j + open_len] == open_delim and open_delim == close_delim:
                    # Same open/close (like $$), alternate
                    depth -= 1
                    if depth == 0:
                        content = text[i + open_len:j]
                        results.append((i, j + close_len - 1, content))
                        i = j + close_len
                        break
                elif text[j:j + close_len] == close_delim:
                    depth -= 1
                    if depth == 0:
                        content = text[i + open_len:j]
                        results.append((i, j + close_len - 1, content))
                        i = j + close_len
                        break
                elif text[j:j + open_len] == open_delim:
                    depth += 1
                j += 1
            else:
                # Unbalanced — skip this opening delimiter
                i += open_len
        else:
            i += 1

    return results


def _find_display_math_regions(text: str) -> List[Tuple[int, int, str, str]]:
    """Find all display math regions ($$...$$ and \\[...\\]).

    Returns list of (start, end, content, delimiter_type) tuples.
    These are always treated as math regardless of content heuristics.
    """
    regions = []

    # $$ ... $$ (use balanced matching to handle nesting correctly)
    for start, end, content in _find_balanced_delimiters(text, "$$", "$$"):
        regions.append((start, end, content, "display"))

    # \[ ... \]
    bracket_open = "\\["
    bracket_close = "\\]"
    i = 0
    while i < len(text):
        idx = text.find(bracket_open, i)
        if idx == -1:
            break
        close_idx = text.find(bracket_close, idx + len(bracket_open))
        if close_idx == -1:
            break
        content = text[idx + len(bracket_open):close_idx]
        regions.append((idx, close_idx + len(bracket_close) - 1, content, "display"))
        i = close_idx + len(bracket_close)

    return regions


def _find_inline_math_regions(text: str) -> List[Tuple[int, int, str, str]]:
    """Find inline math regions (\\(...\\) and $...$).

    For \\(...\\), always treated as math.
    For $...$, applies heuristic filtering to avoid currency false positives.

    Returns list of (start, end, content, delimiter_type) tuples.
    """
    regions = []

    # \( ... \) — always math
    paren_open = "\\("
    paren_close = "\\)"
    i = 0
    while i < len(text):
        idx = text.find(paren_open, i)
        if idx == -1:
            break
        close_idx = text.find(paren_close, idx + len(paren_open))
        if close_idx == -1:
            break
        content = text[idx + len(paren_open):close_idx]
        regions.append((idx, close_idx + len(paren_close) - 1, content, "inline"))
        i = close_idx + len(paren_close)

    # $ ... $ — requires heuristic validation
    # First, collect already-found regions (display math, \[..\], \(..\)) to exclude
    occupied = set()
    for start, end, _, _ in regions:
        for j in range(start, end + 1):
            occupied.add(j)

    # Also mark $$ regions
    for start, end, _, _ in _find_display_math_regions(text):
        for j in range(start, end + 1):
            occupied.add(j)

    i = 0
    while i < len(text):
        if i in occupied:
            i += 1
            continue

        if text[i] == '$' and (i + 1 < len(text) and text[i + 1] != '$'):
            # Potential single-$ opening
            j = i + 1
            while j < len(text):
                if j in occupied:
                    j += 1
                    continue
                if text[j] == '$' and (j + 1 >= len(text) or text[j + 1] != '$'):
                    content = text[i + 1:j]
                    if content.strip() and not any(k in occupied for k in range(i, j + 1)):
                        if _is_likely_latex(content):
                            regions.append((i, j, content, "inline"))
                            # Mark as occupied
                            for k in range(i, j + 1):
                                occupied.add(k)
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1

    return regions


# ---------------------------------------------------------------------------
#  LaTeX content parser / converter
# ---------------------------------------------------------------------------

def _convert_superscript(text: str) -> str:
    """Convert a superscript argument to Unicode superscript."""
    result = text.translate(_SUPERSCRIPT_MAP)
    # If any characters didn't translate, wrap in ^() as fallback
    if result == text and not text.isascii():
        return text
    return result


def _convert_subscript(text: str) -> str:
    """Convert a subscript argument to Unicode subscript."""
    result = text.translate(_SUBSCRIPT_MAP)
    if result == text and not text.isascii():
        return text
    return result


def _extract_brace_arg(text: str, start: int) -> Tuple[str, int]:
    """Extract a brace-delimited argument starting at *start* (which points to '{').

    Handles nesting. Returns (content, end_index) where end_index is the position
    **after** the closing '}', so the caller can continue parsing from that point.
    """
    if start >= len(text) or text[start] != '{':
        return "", start

    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1

    # Unbalanced — return what we have
    return text[start + 1:], len(text)


def _extract_optional_bracket_arg(text: str, start: int) -> Tuple[Optional[str], int]:
    """Extract an optional bracket argument [...] starting at *start*.

    Returns (content_or_None, end_index) where end_index is the position
    **after** the closing ']'.
    """
    if start >= len(text) or text[start] != '[':
        return None, start

    depth = 0
    i = start
    while i < len(text):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1

    return None, start


def _get_next_token(text: str, pos: int) -> Tuple[str, str, int]:
    """Get the next token from *text* starting at *pos*.

    Returns (token_type, token_value, new_pos).
    Types: 'command', 'brace_group', 'char', 'superscript', 'subscript', 'eof'
    """
    if pos >= len(text):
        return ("eof", "", pos)

    ch = text[pos]

    # Backslash command
    if ch == '\\':
        m = _COMMAND_RE.match(text, pos)
        if m:
            return ("command", "\\" + m.group(1), m.end())
        # Escaped character
        if pos + 1 < len(text):
            return ("command", text[pos:pos + 2], pos + 2)
        return ("char", ch, pos + 1)

    # Brace group
    if ch == '{':
        content, end = _extract_brace_arg(text, pos)
        return ("brace_group", content, end)

    # Superscript
    if ch == '^':
        return ("superscript", "^", pos + 1)

    # Subscript
    if ch == '_':
        return ("subscript", "_", pos + 1)

    # Whitespace, &, \\, etc.
    return ("char", ch, pos + 1)


def _convert_latex_content(content: str) -> str:
    """Convert LaTeX math content to Unicode/plaintext.

    This is the core recursive converter. It processes tokens left-to-right,
    applying mappings and handling structured commands like \\frac, \\sqrt, etc.
    """
    result = []
    pos = 0

    while pos < len(content):
        token_type, token_value, pos = _get_next_token(content, pos)

        if token_type == "eof":
            break

        elif token_type == "char":
            # Handle & (alignment separator) and \\ (line break)
            if token_value == '&':
                result.append(" ")
            elif token_value == '\n':
                result.append(" ")
            else:
                result.append(token_value)

        elif token_type == "command":
            cmd = token_value

            # Handle \\ (line break in aligned environments)
            if cmd == '\\\\':
                result.append("\n")
                # Skip optional [spacing] after \\
                opt, new_pos = _extract_optional_bracket_arg(content, pos)
                pos = new_pos
                continue

            # Skip \left, \right (just delimiters, output the next char)
            if cmd in (r"\left", r"\right"):
                continue

            # \frac{num}{den}
            if cmd == r"\frac":
                num, pos = _extract_brace_arg(content, pos) if pos < len(content) and content[pos] == '{' else ("", pos)
                den, pos = _extract_brace_arg(content, pos) if pos < len(content) and content[pos] == '{' else ("", pos)
                num_text = _convert_latex_content(num)
                den_text = _convert_latex_content(den)
                # Use Unicode fraction slash if both are short
                if len(num_text) <= 3 and len(den_text) <= 3:
                    result.append(f"({num_text})/({den_text})")
                else:
                    result.append(f"({num_text})/({den_text})")
                continue

            # \sqrt[n]{x} or \sqrt{x}
            if cmd == r"\sqrt":
                n = None
                if pos < len(content) and content[pos] == '[':
                    n, pos = _extract_optional_bracket_arg(content, pos)
                arg, pos = _extract_brace_arg(content, pos) if pos < len(content) and content[pos] == '{' else ("", pos)
                arg_text = _convert_latex_content(arg)
                if n:
                    result.append(f"({n})√({arg_text})")
                else:
                    result.append(f"√({arg_text})")
                continue

            # \text{...} — render as plain text
            if cmd in FONT_COMMANDS:
                if pos < len(content) and content[pos] == '{':
                    arg, pos = _extract_brace_arg(content, pos)
                    result.append(arg)
                continue

            # \begin{...}...\end{...} — environment (simplified)
            if cmd == r"\begin":
                env_name, pos = _extract_brace_arg(content, pos) if pos < len(content) and content[pos] == '{' else ("", pos)
                # Find matching \end{env_name}
                end_marker = rf"\end{{{env_name}}}"
                end_idx = content.find(end_marker, pos)
                if end_idx != -1:
                    env_body = content[pos:end_idx]
                    result.append(_convert_environment(env_name, env_body))
                    pos = end_idx + len(end_marker)
                else:
                    result.append(f"[{env_name}]")
                continue

            # Simple command mapping
            if cmd in ALL_SIMPLE_MAPS:
                result.append(ALL_SIMPLE_MAPS[cmd])
                continue

            # Unknown command — strip backslash, keep word
            cmd_word = cmd[1:]  # remove leading backslash
            result.append(cmd_word)

        elif token_type == "superscript":
            # Get the next token as the superscript argument
            next_type, next_val, pos = _get_next_token(content, pos)
            if next_type == "brace_group":
                converted = _convert_latex_content(next_val)
                result.append(_convert_superscript(converted))
            elif next_type == "char":
                result.append(_convert_superscript(next_val))
            elif next_type == "command":
                # e.g. ^\prime
                if next_val in ALL_SIMPLE_MAPS:
                    result.append(_convert_superscript(ALL_SIMPLE_MAPS[next_val]))
                else:
                    result.append(_convert_superscript(next_val[1:]))

        elif token_type == "subscript":
            next_type, next_val, pos = _get_next_token(content, pos)
            if next_type == "brace_group":
                converted = _convert_latex_content(next_val)
                result.append(_convert_subscript(converted))
            elif next_type == "char":
                result.append(_convert_subscript(next_val))
            elif next_type == "command":
                if next_val in ALL_SIMPLE_MAPS:
                    result.append(_convert_subscript(ALL_SIMPLE_MAPS[next_val]))
                else:
                    result.append(_convert_subscript(next_val[1:]))

        elif token_type == "brace_group":
            # Bare brace group — just convert its contents
            result.append(_convert_latex_content(token_value))

    return "".join(result)


def _convert_environment(name: str, body: str) -> str:
    """Convert a LaTeX environment (matrix, cases, aligned, etc.) to readable text."""
    name = name.strip().replace("*", "")

    if name in ("pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
                 "matrix", "array", "smallmatrix"):
        return _convert_matrix(body)
    elif name in ("aligned", "align", "align*", "eqnarray", "eqnarray*",
                   "gather", "gather*", "split"):
        return _convert_aligned(body)
    elif name in ("cases", "dcases"):
        return _convert_cases(body)
    else:
        # Generic environment — just convert the body
        return _convert_latex_content(body.strip())


def _convert_matrix(body: str) -> str:
    """Convert matrix environment content to a readable text representation."""
    rows = []
    for row_text in body.split(r"\\"):
        row_text = row_text.strip()
        if not row_text:
            continue
        cells = [_convert_latex_content(c.strip()) for c in row_text.split("&")]
        rows.append("[" + ", ".join(cells) + "]")
    return "[" + "; ".join(rows) + "]"


def _convert_aligned(body: str) -> str:
    """Convert aligned/align environment to newline-separated equations."""
    lines = []
    for line_text in body.split(r"\\"):
        line_text = line_text.strip()
        if not line_text:
            continue
        # Remove alignment &
        line_text = line_text.replace("&", " ")
        lines.append(_convert_latex_content(line_text))
    return "\n".join(lines)


def _convert_cases(body: str) -> str:
    """Convert cases environment to readable conditional form."""
    lines = []
    for line_text in body.split(r"\\"):
        line_text = line_text.strip()
        if not line_text:
            continue
        parts = line_text.split("&")
        converted = [_convert_latex_content(p.strip()) for p in parts]
        lines.append(", ".join(converted))
    return " | ".join(lines)


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def convert_latex_to_plain(text: str) -> str:
    """Convert LaTeX math expressions in *text* to Discord-safe Unicode/plaintext.

    Processing steps:
    1. Extract and protect fenced code blocks and inline code.
    2. Find all math delimiter regions ($$, \\[\\], \\(\\), $).
    3. For each region, convert LaTeX content to Unicode/plaintext.
    4. Restore protected code blocks.

    Args:
        text: The raw LLM response text, potentially containing LaTeX.

    Returns:
        The text with LaTeX converted to Unicode/plaintext.
    """
    if not text or not text.strip():
        return text

    # Step 1: Protect code blocks
    text, fence_placeholders = _extract_blocks(text, _FENCE_RE)
    text, inline_placeholders = _extract_blocks(text, _INLINE_CODE_RE)
    all_placeholders = {**fence_placeholders, **inline_placeholders}

    try:
        # Step 2: Find all math regions
        display_regions = _find_display_math_regions(text)
        inline_regions = _find_inline_math_regions(text)

        all_regions = display_regions + inline_regions

        if not all_regions:
            return _restore_blocks(text, all_placeholders)

        # Sort by start position (descending) so replacements don't shift indices
        all_regions.sort(key=lambda r: r[0], reverse=True)

        # Step 3: Replace each region with converted content
        for start, end, content, delim_type in all_regions:
            try:
                converted = _convert_latex_content(content.strip())
                if not converted.strip():
                    continue

                # For display math, wrap in a code block for better Discord rendering
                if delim_type == "display":
                    # If conversion produced multi-line or complex output, use code block
                    if "\n" in converted or len(converted) > 80:
                        replacement = f"```\n{converted}\n```"
                    else:
                        replacement = f"**{converted}**"
                else:
                    replacement = converted

                text = text[:start] + replacement + text[end + 1:]
            except Exception:
                logger.debug(f"Failed to convert LaTeX region: {content[:100]}", exc_info=True)
                # Leave the original LaTeX in place on failure
                continue

    except Exception:
        logger.warning("Error during LaTeX conversion, returning original text", exc_info=True)

    # Step 4: Restore code blocks
    return _restore_blocks(text, all_placeholders)

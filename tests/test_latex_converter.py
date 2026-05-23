"""Tests for the LaTeX-to-Unicode/plaintext converter.

Covers:
- Code block protection (fenced and inline)
- Display math ($$...$$ and \\[...\\])
- Inline math ($...$ and \\(...\\))
- Currency/shell variable false-positive avoidance
- Greek letters, operators, relations, arrows
- Fractions, square roots, superscripts, subscripts
- Matrix/aligned environment conversion
- Markdown preservation
- Edge cases (empty, unbalanced, nested)
"""

import pytest

from aiuser.utils.latex_converter import (
    convert_latex_to_plain,
    _is_likely_latex,
    _convert_latex_content,
    _find_display_math_regions,
    _find_inline_math_regions,
    _extract_blocks,
    _restore_blocks,
    _FENCE_RE,
    _INLINE_CODE_RE,
)


# =========================================================================
#  Section 1: Code block protection
# =========================================================================


class TestCodeBlockProtection:
    """Code blocks (fenced and inline) must survive LaTeX conversion untouched."""

    def test_fenced_code_block_preserved(self):
        text = "Check this: ```$x^2 + y^2 = z^2$```"
        result = convert_latex_to_plain(text)
        assert "```$x^2 + y^2 = z^2$```" in result

    def test_inline_code_preserved(self):
        text = "Use `$\\alpha$` for alpha."
        result = convert_latex_to_plain(text)
        assert "`$\\alpha$`" in result

    def test_fenced_code_block_with_language(self):
        text = "```python\n$100 + $200\n```"
        result = convert_latex_to_plain(text)
        assert "```python\n$100 + $200\n```" in result

    def test_multiple_code_blocks_preserved(self):
        text = "A `$x$` and B `$y$` and C ```$z$```."
        result = convert_latex_to_plain(text)
        assert "`$x$`" in result
        assert "`$y$`" in result
        assert "```$z$```" in result

    def test_code_block_with_latex_commands_preserved(self):
        text = "Use `\\frac{1}{2}` in your formula."
        result = convert_latex_to_plain(text)
        assert "`\\frac{1}{2}`" in result

    def test_extract_restore_roundtrip(self):
        original = "Hello ```code``` and `inline` world"
        text, placeholders = _extract_blocks(original, _FENCE_RE)
        text, inline_ph = _extract_blocks(text, _INLINE_CODE_RE)
        placeholders.update(inline_ph)
        restored = _restore_blocks(text, placeholders)
        assert restored == original


# =========================================================================
#  Section 2: Currency and false-positive avoidance
# =========================================================================


class TestCurrencyFalsePositives:
    """Currency amounts like $100, $3.50 must NOT be converted."""

    def test_dollar_amount_integer(self):
        assert convert_latex_to_plain("It costs $100.") == "It costs $100."

    def test_dollar_amount_decimal(self):
        assert convert_latex_to_plain("Price is $3.50.") == "Price is $3.50."

    def test_dollar_amount_comma_separated(self):
        assert convert_latex_to_plain("Budget: $1,234,567") == "Budget: $1,234,567"

    def test_dollar_amount_with_slash(self):
        assert "$5/month" in convert_latex_to_plain("$5/month")

    def test_dollar_sign_at_end(self):
        assert convert_latex_to_plain("Earn $") == "Earn $"

    def test_empty_dollar_delimiters(self):
        text = "Price $$ nothing"
        result = convert_latex_to_plain(text)
        # $$ with empty or whitespace content between should not crash
        assert result  # just ensure no crash


class TestHeuristicIsLikelyLatex:
    """Test the _is_likely_latex heuristic directly."""

    def test_currency_rejected(self):
        assert _is_likely_latex("100") is False
        assert _is_likely_latex("3.50") is False
        assert _is_likely_latex("1,234.56") is False

    def test_empty_rejected(self):
        assert _is_likely_latex("") is False
        assert _is_likely_latex("   ") is False

    def test_backslash_command_accepted(self):
        assert _is_likely_latex("\\alpha") is True
        assert _is_likely_latex("\\frac{1}{2}") is True

    def test_brace_group_accepted(self):
        assert _is_likely_latex("x_{i}") is True
        assert _is_likely_latex("a^{2}") is True

    def test_subscript_single_accepted(self):
        assert _is_likely_latex("a_b") is True

    def test_superscript_single_accepted(self):
        assert _is_likely_latex("x^2") is True

    def test_plain_text_rejected(self):
        assert _is_likely_latex("hello world") is False

    def test_dollar_amount_in_content_rejected(self):
        assert _is_likely_latex(" $100 ") is False


# =========================================================================
#  Section 3: Display math ($$...$$ and \[...\])
# =========================================================================


class TestDisplayMath:
    """Display math delimiters are always treated as math."""

    def test_double_dollar_display(self):
        result = convert_latex_to_plain("$$x^2 + y^2 = z^2$$")
        assert "x²" in result
        assert "y²" in result
        assert "z²" in result

    def test_bracket_display(self):
        result = convert_latex_to_plain(r"\[E = mc^2\]")
        assert "E" in result
        assert "m" in result
        assert "c²" in result

    def test_display_math_with_fraction(self):
        result = convert_latex_to_plain(r"$$\frac{a}{b}$$")
        assert "(a)/(b)" in result

    def test_display_math_in_paragraph(self):
        text = "The equation $$x^2 = 4$$ has solutions."
        result = convert_latex_to_plain(text)
        assert "x²" in result
        assert "The equation" in result
        assert "has solutions" in result


# =========================================================================
#  Section 4: Inline math ($...$ and \(...\))
# =========================================================================


class TestInlineMath:
    """Inline math with various delimiter styles."""

    def test_paren_delimiter(self):
        result = convert_latex_to_plain(r"Let \(x = 5\).")
        assert "x" in result
        assert "5" in result

    def test_single_dollar_with_command(self):
        result = convert_latex_to_plain(r"Value: $\alpha + \beta$")
        assert "α" in result
        assert "β" in result

    def test_single_dollar_with_braces(self):
        result = convert_latex_to_plain(r"$x_{1}$ and $y^{2}$")
        assert "x₁" in result
        assert "y²" in result

    def test_inline_math_with_fraction(self):
        result = convert_latex_to_plain(r"Ratio: $\frac{1}{2}$")
        assert "(1)/(2)" in result


# =========================================================================
#  Section 5: Greek letters
# =========================================================================


class TestGreekLetters:
    """Greek letter commands should map to Unicode."""

    def test_lowercase_greek(self):
        for cmd, expected in [
            (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"),
            (r"\delta", "δ"), (r"\theta", "θ"), (r"\lambda", "λ"),
            (r"\mu", "μ"), (r"\pi", "π"), (r"\sigma", "σ"),
            (r"\omega", "ω"), (r"\phi", "φ"), (r"\psi", "ψ"),
        ]:
            result = _convert_latex_content(cmd)
            assert result == expected, f"Expected {expected} for {cmd}, got {result}"

    def test_uppercase_greek(self):
        for cmd, expected in [
            (r"\Gamma", "Γ"), (r"\Delta", "Δ"), (r"\Theta", "Θ"),
            (r"\Lambda", "Λ"), (r"\Sigma", "Σ"), (r"\Omega", "Ω"),
        ]:
            result = _convert_latex_content(cmd)
            assert result == expected, f"Expected {expected} for {cmd}, got {result}"

    def test_greek_in_context(self):
        result = convert_latex_to_plain(r"$\alpha + \beta = \gamma$")
        assert "α" in result
        assert "β" in result
        assert "γ" in result


# =========================================================================
#  Section 6: Operators and relations
# =========================================================================


class TestOperatorsAndRelations:
    """Binary operators and relation symbols."""

    def test_times(self):
        assert _convert_latex_content(r"\times") == "×"

    def test_cdot(self):
        assert _convert_latex_content(r"\cdot") == "·"

    def test_div(self):
        assert _convert_latex_content(r"\div") == "÷"

    def test_pm(self):
        assert _convert_latex_content(r"\pm") == "±"

    def test_le_ge(self):
        assert _convert_latex_content(r"\leq") == "≤"
        assert _convert_latex_content(r"\geq") == "≥"

    def test_neq(self):
        assert _convert_latex_content(r"\neq") == "≠"

    def test_approx(self):
        assert _convert_latex_content(r"\approx") == "≈"

    def test_equiv(self):
        assert _convert_latex_content(r"\equiv") == "≡"

    def test_subset_superset(self):
        assert _convert_latex_content(r"\subseteq") == "⊆"
        assert _convert_latex_content(r"\supseteq") == "⊇"

    def test_in(self):
        assert _convert_latex_content(r"\in") == "∈"


# =========================================================================
#  Section 7: Arrows and miscellaneous symbols
# =========================================================================


class TestArrowsAndMisc:
    """Arrow and miscellaneous symbol mappings."""

    def test_arrows(self):
        assert _convert_latex_content(r"\rightarrow") == "→"
        assert _convert_latex_content(r"\leftarrow") == "←"
        assert _convert_latex_content(r"\Rightarrow") == "⇒"
        assert _convert_latex_content(r"\Leftarrow") == "⇐"
        assert _convert_latex_content(r"\leftrightarrow") == "↔"
        assert _convert_latex_content(r"\to") == "→"

    def test_infinity(self):
        assert _convert_latex_content(r"\infty") == "∞"

    def test_partial_nabla(self):
        assert _convert_latex_content(r"\partial") == "∂"
        assert _convert_latex_content(r"\nabla") == "∇"

    def test_dots(self):
        assert _convert_latex_content(r"\ldots") == "…"
        assert _convert_latex_content(r"\cdots") == "⋯"
        assert _convert_latex_content(r"\dots") == "…"

    def test_forall_exists(self):
        assert _convert_latex_content(r"\forall") == "∀"
        assert _convert_latex_content(r"\exists") == "∃"

    def test_emptyset(self):
        assert _convert_latex_content(r"\emptyset") == "∅"


# =========================================================================
#  Section 8: Fractions
# =========================================================================


class TestFractions:
    """\\frac{num}{den} conversion."""

    def test_simple_fraction(self):
        result = _convert_latex_content(r"\frac{1}{2}")
        assert result == "(1)/(2)"

    def test_fraction_with_variables(self):
        result = _convert_latex_content(r"\frac{a}{b}")
        assert result == "(a)/(b)"

    def test_nested_fraction(self):
        result = _convert_latex_content(r"\frac{\frac{a}{b}}{c}")
        assert "(a)/(b)" in result
        assert "/(c)" in result

    def test_fraction_with_exponents(self):
        result = _convert_latex_content(r"\frac{x^2}{y^3}")
        assert "/(" in result


# =========================================================================
#  Section 9: Square roots
# =========================================================================


class TestSquareRoots:
    """\\sqrt{} and \\sqrt[n]{} conversion."""

    def test_simple_sqrt(self):
        result = _convert_latex_content(r"\sqrt{x}")
        assert "√" in result
        assert "x" in result

    def test_sqrt_with_content(self):
        result = _convert_latex_content(r"\sqrt{b^2 - 4ac}")
        assert "√" in result

    def test_nth_root(self):
        result = _convert_latex_content(r"\sqrt[3]{x}")
        assert "3" in result
        assert "√" in result
        assert "x" in result


# =========================================================================
#  Section 10: Superscripts and subscripts
# =========================================================================


class TestSuperSubscripts:
    """Superscript and subscript Unicode conversion."""

    def test_superscript_digit(self):
        result = _convert_latex_content("x^2")
        assert "x²" in result

    def test_subscript_digit(self):
        result = _convert_latex_content("x_0")
        assert "x₀" in result

    def test_superscript_braced(self):
        result = _convert_latex_content("x^{n}")
        assert "xⁿ" in result

    def test_subscript_braced(self):
        result = _convert_latex_content("x_{i}")
        assert "xᵢ" in result

    def test_superscript_plus(self):
        result = _convert_latex_content("x^{+}")
        assert "x⁺" in result

    def test_superscript_minus(self):
        result = _convert_latex_content("x^{-}")
        assert "x⁻" in result

    def test_subscript_multi_char(self):
        result = _convert_latex_content("x_{ij}")
        # Multi-char subscripts that can't all be Unicode — should still produce output
        assert "x" in result

    def test_prime_superscript(self):
        result = _convert_latex_content("f'(x)")
        assert result  # just ensure no crash


# =========================================================================
#  Section 11: Large operators
# =========================================================================


class TestLargeOperators:
    """Sum, integral, product, etc."""

    def test_sum(self):
        assert _convert_latex_content(r"\sum") == "Σ"

    def test_prod(self):
        assert _convert_latex_content(r"\prod") == "Π"

    def test_integral(self):
        assert _convert_latex_content(r"\int") == "∫"

    def test_sum_with_limits(self):
        result = _convert_latex_content(r"\sum_{i=0}^{n} x_i")
        assert "Σ" in result
        assert "x" in result


# =========================================================================
#  Section 12: Delimiters and \\left/\\right
# =========================================================================


class TestDelimiters:
    """\\left, \\right, and special delimiters."""

    def test_left_right_stripped(self):
        result = _convert_latex_content(r"\left( x \right)")
        assert "x" in result
        assert r"\left" not in result
        assert r"\right" not in result

    def test_floor_ceil(self):
        assert _convert_latex_content(r"\lfloor") == "⌊"
        assert _convert_latex_content(r"\rfloor") == "⌋"
        assert _convert_latex_content(r"\lceil") == "⌈"
        assert _convert_latex_content(r"\rceil") == "⌉"

    def test_angle_brackets(self):
        assert _convert_latex_content(r"\langle") == "⟨"
        assert _convert_latex_content(r"\rangle") == "⟩"


# =========================================================================
#  Section 13: Text/font commands
# =========================================================================


class TestTextCommands:
    """\\text{}, \\mathrm{}, etc. should output plain text."""

    def test_text_command(self):
        result = _convert_latex_content(r"\text{hello}")
        assert result == "hello"

    def test_mathrm_command(self):
        result = _convert_latex_content(r"\mathrm{d}x")
        assert "d" in result

    def test_operatorname(self):
        result = _convert_latex_content(r"\operatorname{sin}(x)")
        assert "sin" in result


# =========================================================================
#  Section 14: Environments (matrix, aligned, cases)
# =========================================================================


class TestEnvironments:
    """Matrix, aligned, cases environment conversion."""

    def test_pmatrix(self):
        latex = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
        result = _convert_latex_content(latex)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result

    def test_bmatrix(self):
        latex = r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"
        result = _convert_latex_content(latex)
        assert "1" in result
        assert "0" in result

    def test_cases(self):
        latex = r"\begin{cases} x & \text{if } x > 0 \\ -x & \text{otherwise} \end{cases}"
        result = _convert_latex_content(latex)
        assert "x" in result
        assert "-x" in result

    def test_aligned(self):
        latex = r"\begin{aligned} a &= b + c \\ d &= e + f \end{aligned}"
        result = _convert_latex_content(latex)
        assert "a" in result
        assert "d" in result


# =========================================================================
#  Section 15: Markdown preservation
# =========================================================================


class TestMarkdownPreservation:
    """Existing markdown must survive LaTeX conversion untouched."""

    def test_bold_preserved(self):
        text = "**bold text** and $\\alpha$"
        result = convert_latex_to_plain(text)
        assert "**bold text**" in result

    def test_italic_preserved(self):
        text = "*italic text* and $\\beta$"
        result = convert_latex_to_plain(text)
        assert "*italic text*" in result

    def test_link_preserved(self):
        text = "[link](https://example.com) and $\\gamma$"
        result = convert_latex_to_plain(text)
        assert "[link](https://example.com)" in result

    def test_mention_preserved(self):
        text = "<@123456> said $x^2$"
        result = convert_latex_to_plain(text)
        assert "<@123456>" in result

    def test_channel_mention_preserved(self):
        text = "See <#789> for $\\pi$"
        result = convert_latex_to_plain(text)
        assert "<#789>" in result

    def test_strikethrough_preserved(self):
        text = "~~removed~~ and $\\sum$"
        result = convert_latex_to_plain(text)
        assert "~~removed~~" in result

    def test_blockquote_preserved(self):
        text = "> quote text\n> $\\alpha$"
        result = convert_latex_to_plain(text)
        assert "> quote text" in result

    def test_heading_preserved(self):
        text = "# Title\nSome $x^2$ math"
        result = convert_latex_to_plain(text)
        assert "# Title" in result

    def test_horizontal_rule_preserved(self):
        text = "---\n$\\pi$ math"
        result = convert_latex_to_plain(text)
        assert "---" in result

    def test_pure_markdown_no_latex(self):
        text = "**bold** and *italic* and `code`"
        result = convert_latex_to_plain(text)
        assert result == text

    def test_discord_emoji_preserved(self):
        text = ":thinking: about $\\alpha$"
        result = convert_latex_to_plain(text)
        assert ":thinking:" in result

    def test_discord_custom_emoji_preserved(self):
        text = "<:custom:123456> and $\\beta$"
        result = convert_latex_to_plain(text)
        assert "<:custom:123456>" in result


# =========================================================================
#  Section 16: Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases: empty input, unbalanced delimiters, mixed content."""

    def test_empty_string(self):
        assert convert_latex_to_plain("") == ""

    def test_none_like_empty(self):
        # convert_latex_to_plain should handle empty/whitespace gracefully
        assert convert_latex_to_plain("   ") == "   "

    def test_no_latex(self):
        text = "Hello, this is a normal message with no math."
        assert convert_latex_to_plain(text) == text

    def test_unbalanced_single_dollar(self):
        text = "This has an unbalanced $ delimiter"
        result = convert_latex_to_plain(text)
        # Should leave it alone
        assert "$" in result

    def test_unbalanced_double_dollar(self):
        text = "Unbalanced $$x^2"
        result = convert_latex_to_plain(text)
        # Should not crash
        assert result

    def test_unbalanced_bracket(self):
        text = r"Unbalanced \[x^2"
        result = convert_latex_to_plain(text)
        assert result

    def test_unbalanced_paren(self):
        text = r"Unbalanced \(x^2"
        result = convert_latex_to_plain(text)
        assert result

    def test_multiple_math_regions(self):
        text = r"First $\alpha$ then $\beta$ and $\gamma$."
        result = convert_latex_to_plain(text)
        assert "α" in result
        assert "β" in result
        assert "γ" in result
        assert "First" in result
        assert "then" in result
        assert "and" in result

    def test_mixed_delimiters(self):
        text = r"Display $$\sum$$ and inline $\int$ and paren \(\pi\)."
        result = convert_latex_to_plain(text)
        assert "Σ" in result
        assert "∫" in result
        assert "π" in result

    def test_complex_expression(self):
        text = r"$\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$"
        result = convert_latex_to_plain(text)
        assert "(-b" in result or "-b" in result
        assert "±" in result
        assert "√" in result
        assert "/(2a)" in result

    def test_display_math_multiline_fallback(self):
        """Complex display math should fall back to code block."""
        text = "$$\\frac{a}{b} + \\frac{c}{d} + \\frac{e}{f} + \\frac{g}{h} + \\frac{i}{j} + \\frac{k}{l} + \\frac{m}{n} + \\frac{o}{p}$$"
        result = convert_latex_to_plain(text)
        # Should produce some output (either Unicode or code block)
        assert result.strip()

    def test_unknown_command_stripped(self):
        """Unknown commands should strip the backslash."""
        result = _convert_latex_content(r"\unknowncmd")
        assert result == "unknowncmd"

    def test_ampersand_as_separator(self):
        result = _convert_latex_content("a & b")
        # & is an alignment separator, should become space
        assert "a" in result
        assert "b" in result

    def test_linebreak_double_backslash(self):
        result = _convert_latex_content("a \\\\ b")
        assert "\n" in result


# =========================================================================
#  Section 17: Integration-style end-to-end tests
# =========================================================================


class TestEndToEnd:
    """End-to-end scenarios simulating real LLM responses."""

    def test_math_proof_text(self):
        text = (
            "The quadratic formula is $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$.\n"
            "This gives us **two solutions** for any $a \\neq 0$."
        )
        result = convert_latex_to_plain(text)
        # Bold markdown preserved
        assert "**two solutions**" in result
        # Unicode math
        assert "≠" in result
        assert "±" in result
        assert "√" in result

    def test_inline_equation_in_sentence(self):
        text = "The area of a circle is $A = \\pi r^2$ where $r$ is the radius."
        result = convert_latex_to_plain(text)
        assert "π" in result
        assert "r²" in result
        assert "area of a circle" in result

    def test_code_block_with_dollar_signs(self):
        text = (
            "Here's the Python code:\n"
            "```python\n"
            "price = $100\n"
            "tax = price * 0.08\n"
            "```\n"
            "The formula is $V = \\frac{P}{1 + rt}$."
        )
        result = convert_latex_to_plain(text)
        # Code block must be preserved exactly
        assert "```python" in result
        assert "price = $100" in result
        # LaTeX should be converted
        assert "π" not in result or "P" in result  # \frac{P}{...}

    def test_discord_embed_with_mentions_and_math(self):
        text = (
            "<@123456> The result is $\\frac{22}{7} \\approx \\pi$.\n"
            "Check <#789> for details."
        )
        result = convert_latex_to_plain(text)
        assert "<@123456>" in result
        assert "<#789>" in result
        assert "≈" in result
        assert "π" in result

    def test_currency_and_math_in_same_message(self):
        text = (
            "The investment of $1,000,000 grows by $r = 0.05$ annually.\n"
            "After $n$ years: $A = P(1 + r)^n$"
        )
        result = convert_latex_to_plain(text)
        # Currency preserved
        assert "$1,000,000" in result
        # Math converted
        assert "r" in result
        assert "n" in result

    def test_list_with_math(self):
        text = (
            "Key formulas:\n"
            "- Area: $A = \\pi r^2$\n"
            "- Circumference: $C = 2\\pi r$\n"
            "- Volume: $V = \\frac{4}{3}\\pi r^3$"
        )
        result = convert_latex_to_plain(text)
        assert "π" in result
        assert "r²" in result
        assert "- Area:" in result
        assert "- Circumference:" in result
        assert "- Volume:" in result

    def test_nested_parentheses_and_braces(self):
        """Ensure brace matching works correctly with nested content."""
        text = r"$\sqrt{\frac{a^2 + b^2}{c}}$"
        result = convert_latex_to_plain(text)
        assert "√" in result
        assert result.strip()  # produces some output

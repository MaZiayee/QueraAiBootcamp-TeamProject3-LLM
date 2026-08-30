"""Shared text utilities for the Digikala assistant project.

Both notebooks import from here so that the exact same normalisation is applied
to the corpus at index time and to the user's query at search time. If these
two ever diverge, lexical matching silently fails.
"""

import re

__all__ = ["normalize_text", "PLACEHOLDER_BRAND"]

# Single-pass character mapping. maketrans builds a lookup table, so every rule
# below costs one scan of the string rather than one scan per rule.
_TRANSLATION = str.maketrans({
    # Arabic letter forms -> Persian equivalents. These render identically but
    # are different code points, so leaving them silently breaks all matching.
    "\u064A": "\u06CC",   # ARABIC YEH         -> PERSIAN YEH
    "\u0649": "\u06CC",   # ALEF MAKSURA       -> PERSIAN YEH
    "\u0643": "\u06A9",   # ARABIC KAF         -> PERSIAN KEHEH
    "\u0629": "\u0647",   # TEH MARBUTA        -> HEH
    "\u06C0": "\u0647",   # HEH WITH YEH ABOVE -> HEH
    "\u0623": "\u0627",   # ALEF + HAMZA ABOVE -> ALEF
    "\u0625": "\u0627",   # ALEF + HAMZA BELOW -> ALEF
    "\u0671": "\u0627",   # ALEF WASLA         -> ALEF
    # Both Arabic-Indic digit ranges -> ASCII, so numeric tokens compare equal.
    **{chr(0x0660 + d): str(d) for d in range(10)},
    **{chr(0x06F0 + d): str(d) for d in range(10)},
    # Punctuation
    "\u060C": ",", "\u061B": ";", "\u061F": "?", "\u066A": "%",
    # Invisible characters that carry no meaning and only break tokenisation.
    # Note: ZWNJ (U+200C) is deliberately absent; it is meaningful in Persian.
    "\u200B": "", "\u200D": "", "\u200E": "", "\u200F": "", "\uFEFF": "",
})

# Harakat (short-vowel marks) and tatweel (decorative letter stretching).
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")
# Letters repeated 3+ times: "عالییییی" -> "عالی". Digits are excluded on
# purpose: without that guard a price like 1000000 would collapse to 10.
_ELONGATION = re.compile(r"([^\W\d_])\1{2,}")
# Keep the ZWNJ but drop any spaces users typed around it.
_ZWNJ = re.compile(r"[ \t]*\u200C[ \t]*")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalise one Persian string for matching, embedding and tokenisation.

    The output is for machine comparison only. Callers must keep the original
    text for display, because quoting a review back to a user has to show what
    that user actually wrote.

    Note: this uses Python's `re` module directly. pandas 3 stores strings via
    PyArrow and routes `.str.replace` to RE2, which supports neither the
    backreference in _ELONGATION nor unicode code-point escapes.
    """
    text = text.translate(_TRANSLATION)
    text = _DIACRITICS.sub("", text)
    text = _ELONGATION.sub(r"\1", text)
    text = _ZWNJ.sub("\u200c", text)
    return _WHITESPACE.sub(" ", text).strip()


# The catalogue uses this word instead of leaving the brand empty. It marks
# ~56% of products, so it must never enter an embedding or a brand filter.
PLACEHOLDER_BRAND = normalize_text("متفرقه")

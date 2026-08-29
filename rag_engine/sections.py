"""Shared section-number regex logic.

`ingest.py` uses SECTION_TAG_PATTERN to tag each chunk with the section it
belongs to. The LangGraph `chain` node reuses the SAME parsing (via
`find_section_references`) to detect when a retrieved chunk points at *another*
section, so it can pull that one in too.
"""

import re

# Used by ingest.py: a chunk that starts a new section looks like "33." or
# "Section 33." at/near the top of the page.
SECTION_TAG_PATTERN = re.compile(r"(?:Section\s+)?(\d+)\.", re.IGNORECASE)

# Used by the chain node: an in-body cross reference, e.g.
#   "... in accordance with section 143 of this Constitution ..."
#   "... Nothing in sections 37, 38, 39, 40 and 41 ..."
# Capture the "section(s)" keyword plus the run of numbers/commas/"and" after it.
_SECTION_REF_PATTERN = re.compile(
    r"\bsections?\s+((?:\d{1,3})(?:\s*(?:,|and)\s*\d{1,3})*)",
    re.IGNORECASE,
)

# The 1999 Constitution has 320 numbered sections; anything past that in a
# "section N" match is almost certainly a stray number, not a real reference.
MAX_SECTION_NUMBER = 320


def tag_section(text: str, current: str = "Preamble") -> str:
    """Return the section label for a chunk (mirrors ingest.py behaviour)."""
    m = SECTION_TAG_PATTERN.search(text or "")
    return f"Section {m.group(1)}" if m else current


def find_section_references(text: str, exclude: set[str] | None = None) -> list[str]:
    """Section labels referenced *inside* `text`, e.g. ["Section 37", "Section 38"].

    Handles enumerations ("sections 37, 38, 39, 40 and 41"), dedupes, drops
    out-of-range numbers, and drops anything in `exclude` (the sections we
    already retrieved).
    """
    exclude = exclude or set()
    found: list[str] = []
    seen: set[str] = set()
    for run in _SECTION_REF_PATTERN.findall(text or ""):
        for num in re.findall(r"\d{1,3}", run):
            n = int(num)
            if not (1 <= n <= MAX_SECTION_NUMBER):
                continue
            label = f"Section {n}"
            if label in exclude or label in seen:
                continue
            seen.add(label)
            found.append(label)
    return found

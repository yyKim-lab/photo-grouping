"""UI translation strings — the app's own menus/buttons/labels, in the
language set by the ui_language app setting (see repository.LANGUAGES).

Distinct from autobio.py's narrative_language: that controls what
language *diary text* is written in, not the interface around it — someone
can read the app in English while writing their diary in Korean, or the
reverse.

Keys are "page.name" dotted strings, one dict per language, all keyed the
same way. A language missing a key falls back to English, then to the raw
key itself — so an incomplete translation degrades to English for just
that string rather than breaking the page. Every language dict is meant
to have the same keys as EN (the reference set); new keys should be added
to all six at once to avoid silent fallback.

Count phrases (e.g. "12 photos") embed the number via a "{n}" placeholder
rather than being built by hand in the template — see t()'s **kwargs
formatting. English/Spanish/French additionally get a "_one" variant for
the count==1 case (real singular/plural agreement); Korean/Japanese don't
inflect for count at all, and Ukrainian's real plural rule (one/few/many)
is approximated as a single invariant form rather than fully implemented —
a deliberate simplification, not an oversight.
"""

from __future__ import annotations

import json
from pathlib import Path

# Source of truth lives in locales/*.json (one file per language, English
# is authoritative — see crowdin.yml / README.md "Translation management
# (Crowdin)"). Loaded once at import time into the same TRANSLATIONS shape
# this module has always exposed, so nothing downstream (t(), the tests,
# the context processor) needs to change for the Crowdin migration.
_LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_translations() -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    for path in sorted(_LOCALES_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            tables[path.stem] = json.load(f)
    return tables


TRANSLATIONS: dict[str, dict[str, str]] = _load_translations()

def t(key: str, lang: str, count: int | None = None, **kwargs) -> str:
    """Looks up `key` in `lang`'s table, falling back to English, then to
    the raw key itself (so a missing translation degrades visibly rather
    than crashing). If `count` is given and a "{key}_one" entry exists,
    that's used instead for the count==1 case (see module docstring)."""
    table = TRANSLATIONS.get(lang) or TRANSLATIONS["en"]
    lookup_key = key
    if count == 1:
        one_key = f"{key}_one"
        if one_key in table or one_key in TRANSLATIONS["en"]:
            lookup_key = one_key
    text = table.get(lookup_key) or TRANSLATIONS["en"].get(lookup_key) or TRANSLATIONS["en"].get(key) or key
    if count is not None:
        kwargs.setdefault("n", count)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass  # a malformed/missing placeholder shouldn't crash the page
    return text

"""Shared test helper: extract an HTML attribute value the way a real
browser would (not via substring matching), so tests can catch attribute
values that get truncated by an unescaped embedded quote — e.g. `tojson`
output placed inside a double-quoted attribute without `| forceescape`,
which silently cuts the attribute short at the JSON string's own quote.
"""

import json
import re
from html.parser import HTMLParser


class _AttrFinder(HTMLParser):
    def __init__(self, css_class: str, attr_name: str):
        super().__init__(convert_charrefs=True)
        self.css_class = css_class
        self.attr_name = attr_name
        self.found = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if self.css_class in classes and self.attr_name in attrs:
            self.found = attrs[self.attr_name]


def get_attr_by_class(html: str, css_class: str, attr_name: str) -> str:
    """Parse `html` and return the value of `attr_name` on the first
    element carrying `css_class`, decoded the way a browser would
    (entities resolved, value not truncated at an embedded quote).
    Raises AssertionError if no such element/attribute is found."""
    parser = _AttrFinder(css_class, attr_name)
    parser.feed(html)
    assert parser.found is not None, (
        f"no element with class={css_class!r} carrying attr={attr_name!r} found"
    )
    return parser.found


class _FormFinder(HTMLParser):
    def __init__(self, action_substring: str):
        super().__init__(convert_charrefs=True)
        self.action_substring = action_substring
        self.found = None

    def handle_starttag(self, tag, attrs):
        if tag != "form":
            return
        attrs = dict(attrs)
        if self.action_substring in (attrs.get("action") or ""):
            self.found = attrs.get("onsubmit")


def get_form_onsubmit_by_action(html: str, action_substring: str) -> str:
    """Parse `html` and return the `onsubmit` attribute of the first
    <form> whose `action` contains `action_substring`, decoded the way a
    browser would (not truncated at an embedded quote). Raises
    AssertionError if no such form/attribute is found."""
    parser = _FormFinder(action_substring)
    parser.feed(html)
    assert parser.found is not None, (
        f"no <form action*={action_substring!r}> with an onsubmit attr found"
    )
    return parser.found


def decode_confirm_message(onsubmit: str) -> str:
    """Given a well-formed `return confirm("...");` attribute value (as
    returned by get_form_onsubmit_by_action), decode the embedded JSON
    string — including any `\\uXXXX` escapes `tojson` emits for non-ASCII
    text — back to the real message text, the way a JS engine would when
    the handler actually runs."""
    match = re.fullmatch(r"return confirm\((\".*\")\);", onsubmit, flags=re.S)
    assert match is not None, f"not a well-formed confirm() onsubmit: {onsubmit!r}"
    return json.loads(match.group(1))

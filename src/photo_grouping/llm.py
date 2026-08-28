"""Minimal client for LLM APIs — §4.6's Autobio narrative drafting.

Supports Anthropic and OpenAI. Which one runs is picked automatically by
which API key is actually configured — some people have one, some the
other, and neither autobio.py nor the web routes need to know or care
which; they just call complete(). AUTOBIO_LLM_PROVIDER forces a specific
choice if both happen to be configured and a particular one is wanted.

Stdlib-only (urllib) for both, matching this project's existing pattern
for every other external API (google_auth.py, picker_client.py,
geocoding.py): no SDK dependency for what's fundamentally one JSON POST
each.

API key resolution order (same shape for both providers):
  1. <PROVIDER>_API_KEY environment variable (matches each provider's own
     SDK convention, so a key already exported for other tools just works).
  2. secrets/<provider>_api_key.txt (first line, whitespace-stripped) —
     same "local secrets folder" convention as client_secret.json/
     token.json for Google, for a key that's easier to just save to a
     file than export in every shell.

Endpoint/format confirmed against each provider's docs
(https://docs.anthropic.com/en/api/messages,
https://platform.openai.com/docs/api-reference/chat) as of 2026-08 —
re-verify if requests start failing, same caveat this codebase already
notes for Google's APIs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Sonnet: good quality/cost balance for narrative drafting — this isn't a
# coding task needing Opus, and Haiku's prose is noticeably flatter for
# the kind of writing §4.6 wants. gpt-4o is OpenAI's rough equivalent
# pick for the same reason. Both overridable per-deployment without a
# code change.
DEFAULT_MODEL = os.environ.get("AUTOBIO_MODEL", "claude-sonnet-5")  # kept for backward compat
DEFAULT_ANTHROPIC_MODEL = os.environ.get("AUTOBIO_ANTHROPIC_MODEL", DEFAULT_MODEL)
DEFAULT_OPENAI_MODEL = os.environ.get("AUTOBIO_OPENAI_MODEL", "gpt-4o")
DEFAULT_MAX_TOKENS = 4096


class LLMNotConfigured(RuntimeError):
    """No usable API key found for any provider. Raised rather than
    silently degrading, since a narrative that's supposed to exist and
    doesn't is a very different failure than the app's other "advisory
    hint" fallbacks (geocoding, OCR) — Autobio's whole point is the draft
    text, so there's nothing sensible to fall back to."""


def _secrets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "secrets"


def _read_key_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    content = path.read_text().strip()
    return content.splitlines()[0].strip() if content else None


def resolve_api_key(key_path: Optional[Path] = None) -> str:
    """Anthropic-specific resolver — kept under its original name/signature
    (predates OpenAI support) since it's a documented entry point on its
    own, not just an internal helper for complete()."""
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key.strip()

    path = key_path or (_secrets_dir() / "anthropic_api_key.txt")
    found = _read_key_file(path)
    if found:
        return found

    raise LLMNotConfigured(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY, or save one to "
        f"{path} — see README.md 'Autobio setup'."
    )


def resolve_openai_key(key_path: Optional[Path] = None) -> Optional[str]:
    """OpenAI counterpart to resolve_api_key(). Returns None rather than
    raising when unconfigured — unlike resolve_api_key(), this is only
    ever consulted as one candidate among several (see
    resolve_provider_and_key()), where "not this one" isn't yet an error."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    path = key_path or (_secrets_dir() / "openai_api_key.txt")
    return _read_key_file(path)


def resolve_provider_and_key(
    *, anthropic_key_path: Optional[Path] = None, openai_key_path: Optional[Path] = None
) -> tuple[str, str]:
    """Picks which provider to use and returns (provider, key).

    AUTOBIO_LLM_PROVIDER ("anthropic" | "openai") forces a specific
    choice — useful if both keys happen to be configured and a specific
    one is wanted. Otherwise: Anthropic wins if configured (this app's
    original, spec-following default), else OpenAI, else a clear error
    naming both options rather than silently picking neither.

    The two `*_key_path` overrides exist for tests — same reason
    resolve_api_key() takes one — so this can be exercised without
    touching whatever's actually saved in secrets/ on the machine running
    the tests.
    """
    forced = (os.environ.get("AUTOBIO_LLM_PROVIDER") or "").strip().lower()
    if forced == "anthropic":
        return "anthropic", resolve_api_key(anthropic_key_path)
    if forced == "openai":
        key = resolve_openai_key(openai_key_path)
        if not key:
            raise LLMNotConfigured(
                "AUTOBIO_LLM_PROVIDER=openai but no OpenAI API key is configured. "
                "Set OPENAI_API_KEY, or save one to "
                f"{openai_key_path or _secrets_dir() / 'openai_api_key.txt'}."
            )
        return "openai", key
    if forced:
        raise LLMNotConfigured(
            f"Unknown AUTOBIO_LLM_PROVIDER: {forced!r} (expected 'anthropic' or 'openai')."
        )

    try:
        return "anthropic", resolve_api_key(anthropic_key_path)
    except LLMNotConfigured:
        pass
    openai_key = resolve_openai_key(openai_key_path)
    if openai_key:
        return "openai", openai_key

    raise LLMNotConfigured(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or save "
        f"one to {anthropic_key_path or _secrets_dir() / 'anthropic_api_key.txt'} or "
        f"{openai_key_path or _secrets_dir() / 'openai_api_key.txt'} — see README.md 'Autobio setup'."
    )


def _complete_anthropic(prompt: str, *, system: Optional[str], model: str, max_tokens: int, api_key: str) -> str:
    body: dict = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system

    request = urllib.request.Request(ANTHROPIC_ENDPOINT, data=json.dumps(body).encode(), method="POST")
    request.add_header("x-api-key", api_key)
    request.add_header("anthropic-version", ANTHROPIC_VERSION)
    request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"Anthropic API error {e.code}: {detail}") from e

    return "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )


def _complete_openai(prompt: str, *, system: Optional[str], model: str, max_tokens: int, api_key: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "max_completion_tokens": max_tokens, "messages": messages}

    request = urllib.request.Request(OPENAI_ENDPOINT, data=json.dumps(body).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI API error {e.code}: {detail}") from e

    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content") or ""


def complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """One-shot completion: sends a single user (+ optional system)
    message, returns the response text. No conversation state — each
    Autobio generation is a fresh, independent call.

    Provider selection: if `provider` isn't given, an explicit `api_key`
    means "use Anthropic with this key" (the original, pre-multi-provider
    behavior — what every existing caller/test that passes api_key=...
    already expects). With neither given, the provider is auto-detected
    from whichever key is actually configured, via
    resolve_provider_and_key().
    """
    if provider is None:
        if api_key is not None:
            provider = "anthropic"
        else:
            provider, api_key = resolve_provider_and_key()
    elif api_key is None:
        api_key = resolve_api_key() if provider == "anthropic" else resolve_openai_key()
        if not api_key:
            raise LLMNotConfigured(f"No API key configured for provider {provider!r}.")

    if provider == "anthropic":
        return _complete_anthropic(
            prompt, system=system, model=model or DEFAULT_ANTHROPIC_MODEL, max_tokens=max_tokens, api_key=api_key
        )
    if provider == "openai":
        return _complete_openai(
            prompt, system=system, model=model or DEFAULT_OPENAI_MODEL, max_tokens=max_tokens, api_key=api_key
        )
    raise LLMNotConfigured(f"Unknown provider: {provider!r} (expected 'anthropic' or 'openai').")

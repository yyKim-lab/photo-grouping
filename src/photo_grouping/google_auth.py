"""Google OAuth 2.0 for a locally-run app (loopback redirect + PKCE).

Uses the "loopback IP address" flow Google documents for installed/desktop
apps: https://developers.google.com/identity/protocols/oauth2/native-app
We spin up a throwaway local HTTP server on an OS-assigned port, send the
user to Google's consent screen with that port baked into redirect_uri, and
catch the single redirect it sends back. PKCE is used because a desktop
client's "client_secret" isn't actually secret (it ships in a file on the
user's machine) — PKCE is what makes the exchange safe regardless.

Stdlib only: urllib for HTTP, http.server for the loopback catcher.

Scope and endpoints confirmed against Google's Photos Picker API docs
(https://developers.google.com/photos/picker/*) as of 2026-08 — per the
spec's own warning (§1, §6), re-verify these if auth starts failing, since
Google does change scope names/endpoints over time.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
# "openid email" added (on top of the Picker scope this app actually
# needs) purely so the token exchange's id_token carries the signed-in
# account's email — Settings' "Connect your accounts" page shows it so a
# self-hoster can tell *which* Google account is connected without
# leaving the app. Doesn't grant access to anything beyond the account's
# email address and a stable subject id.
SCOPE = "openid email https://www.googleapis.com/auth/photospicker.mediaitems.readonly"


@dataclass
class ClientCredentials:
    client_id: str
    client_secret: str


def load_client_credentials(path: Path) -> ClientCredentials:
    """Reads the JSON Google Cloud Console gives you when you create an
    OAuth client of type "Desktop app" (§6 build-order step 1). That file
    is a credential — keep it out of version control (see .gitignore)."""
    data = json.loads(path.read_text())
    block = data.get("installed") or data.get("web")
    if not block:
        raise ValueError(f"Unrecognized client secret file format: {path}")
    return ClientCredentials(client_id=block["client_id"], client_secret=block["client_secret"])


def _email_from_id_token(id_token: str) -> str | None:
    """Pulls the `email` claim out of the id_token's JWT payload — no
    signature verification, deliberately: this token just arrived
    directly from Google's own token endpoint over TLS in the same
    exchange, so there's no untrusted third party to verify against
    (unlike, say, an id_token handed to us by a browser redirect, which
    would need real verification). Purely cosmetic display use — never
    used for anything access-control-related."""
    try:
        payload_b64 = id_token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return payload.get("email")
    except (IndexError, ValueError, UnicodeDecodeError):
        return None


def get_cached_email(token_cache_path: Path) -> str | None:
    """Read-only: whatever email get_access_token() last cached, or None
    if never signed in (or signed in before the `email` scope was added —
    degrades gracefully, no re-auth forced just to keep working)."""
    if not token_cache_path.exists():
        return None
    try:
        return json.loads(token_cache_path.read_text()).get("email")
    except (json.JSONDecodeError, OSError):
        return None


def _make_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
        self.server.result = params  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body>Signed in. You can close this tab and return to the terminal.</body></html>"
        )

    def log_message(self, format, *args):  # noqa: A002 - matches BaseHTTPRequestHandler signature
        pass  # keep the spike script's stdout clean


def _run_loopback_flow(creds: ClientCredentials) -> dict:
    server = http.server.HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    server.result = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/"

    verifier, challenge = _make_pkce_pair()
    state = secrets.token_urlsafe(16)

    auth_url = AUTH_ENDPOINT + "?" + urllib.parse.urlencode(
        {
            "client_id": creds.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )

    print(f"Opening browser for Google sign-in:\n{auth_url}\n")
    webbrowser.open(auth_url)

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout=300)
    result = server.result
    server.server_close()

    if result is None:
        raise TimeoutError("Timed out waiting for the OAuth redirect (5 min).")
    if result.get("state") != state:
        raise ValueError("OAuth state mismatch on redirect — possible interference, aborting.")
    if "error" in result:
        raise RuntimeError(f"Google OAuth error: {result['error']}")

    token_request = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=urllib.parse.urlencode(
            {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "code": result["code"],
                "code_verifier": verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }
        ).encode(),
        method="POST",
    )
    with urllib.request.urlopen(token_request) as resp:
        return json.loads(resp.read())


def _refresh(creds: ClientCredentials, refresh_token: str) -> dict:
    request = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=urllib.parse.urlencode(
            {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode(),
        method="POST",
    )
    with urllib.request.urlopen(request) as resp:
        return json.loads(resp.read())


def get_access_token(creds: ClientCredentials, token_cache_path: Path) -> str:
    """Returns a valid access token. Runs the interactive consent flow (opens
    a browser) only the first time, or after a refresh token stops working;
    every other call refreshes silently from the cached refresh_token."""
    token_cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict = {}
    if token_cache_path.exists():
        cached = json.loads(token_cache_path.read_text())

    if cached.get("refresh_token"):
        try:
            refreshed = _refresh(creds, cached["refresh_token"])
            cached["access_token"] = refreshed["access_token"]
            token_cache_path.write_text(json.dumps(cached))
            return cached["access_token"]
        except urllib.error.HTTPError:
            pass  # refresh token is dead — fall through to a fresh interactive flow

    tokens = _run_loopback_flow(creds)
    if "refresh_token" not in tokens and cached.get("refresh_token"):
        tokens["refresh_token"] = cached["refresh_token"]
    # id_token is only returned on this fresh-consent path, not on a plain
    # refresh (see the branch above) — resolve email here and carry it
    # forward on every future refresh via `cached`, same as refresh_token
    # itself above. Falls back to whatever was cached before rather than
    # clearing it, so a same-account re-auth that (for whatever reason)
    # doesn't include an id_token this time doesn't blank out a
    # previously-known email.
    email = _email_from_id_token(tokens["id_token"]) if "id_token" in tokens else None
    tokens["email"] = email or cached.get("email")
    token_cache_path.write_text(json.dumps(tokens))
    return tokens["access_token"]

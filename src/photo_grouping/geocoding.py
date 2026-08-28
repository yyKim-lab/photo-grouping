"""Reverse geocoding — turns a LocationCluster centroid into a suggested
place name (§3 step 6, §6 step 6) — and, further down, the reverse
direction: forward geocoding a typed place name into coordinates, for
§4.4 photo-location editing (see forward_geocode()).

Routes by region, because no single provider is good everywhere and the
good regional ones are free:

  Korea      Kakao Local      best Korean POI; free key, no card
  Japan      Yahoo! JAPAN     YOLP; 50,000 req/day free
  elsewhere  Nominatim (OSM)  free, no key at all

Providers are tried in order and each may decline (no key configured, or no
result for these coordinates), falling through to the next. Nominatim is
last and needs no configuration, so this always returns *something* out of
the box and simply gets better as keys are added.

Why regional providers rather than one global one: OSM's Korean POI
coverage is thin, so Nominatim answers with administrative addresses like
"방일리, 설악면, 가평군, 경기도" where Kakao can answer with the actual venue.
That difference is the whole point of §3's "suggested place name".

Note on hosted OSM services (geocode.maps.co and similar): they serve the
same Nominatim/OSM data, so they return the same administrative addresses.
They buy rate limit headroom, not better names — and §3 geocodes once per
cluster, batched, so the limit isn't the binding constraint here.

Configuration is by environment variable, so keys never land in the repo:
    KAKAO_REST_API_KEY   from developers.kakao.com
    YAHOO_JP_CLIENT_ID   from developer.yahoo.co.jp
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

# Must stay ASCII: HTTP headers are latin-1 encoded, so typographic
# characters raise UnicodeEncodeError at request time — a failure mocked
# tests can't catch, since it happens inside http.client.
USER_AGENT = "photo-grouping-app (personal use; see spec section 7)"

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_ENDPOINT = "https://nominatim.openstreetmap.org/search"
KAKAO_COORD2ADDRESS = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
KAKAO_CATEGORY_SEARCH = "https://dapi.kakao.com/v2/local/search/category.json"
KAKAO_KEYWORD_SEARCH = "https://dapi.kakao.com/v2/local/search/keyword.json"
YAHOO_JP_REVERSE = "https://map.yahooapis.jp/geoapi/V1/reverseGeoCoder"
YAHOO_JP_GEOCODER = "https://map.yahooapis.jp/geocode/V1/geoCoder"

# Nominatim's usage policy caps at 1 request/second and requires a
# descriptive User-Agent. The regional providers are far more generous, but
# one shared limiter keeps this polite everywhere.
MIN_INTERVAL_SECONDS = 1.0
_last_request_time: float = -1000.0

# Rough bounding boxes, used only to decide which provider to *try* first.
# They overlap around the Korea Strait, which is fine: a provider that has
# nothing for a coordinate declines and the next one is tried, so a wrong
# guess costs one request rather than a wrong answer.
KOREA_BBOX = (33.0, 38.7, 124.5, 132.0)  # lat_min, lat_max, lng_min, lng_max
JAPAN_BBOX = (24.0, 46.0, 122.9, 154.0)


def _in_bbox(lat: float, lng: float, bbox: tuple) -> bool:
    lat_min, lat_max, lng_min, lng_max = bbox
    return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max


def _respect_rate_limit() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _get_json(url: str, headers: Optional[dict] = None) -> Optional[dict]:
    _respect_rate_limit()
    request = urllib.request.Request(url)
    request.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        # A provider being unreachable, unauthorized, or over quota should
        # fall through to the next one rather than fail the whole run — a
        # place-name hint is advisory (§3 step 6), never load-bearing.
        return None


# ---------------------------------------------------------------------
# Providers. Each returns a place name, or None to decline.
# ---------------------------------------------------------------------


def kakao(lat: float, lng: float) -> Optional[str]:
    """Kakao Local. Tries a nearby POI first, since a venue name is what
    §3 actually wants, and falls back to the address if there's no POI
    close enough."""
    key = os.environ.get("KAKAO_REST_API_KEY")
    if not key or not _in_bbox(lat, lng, KOREA_BBOX):
        return None
    headers = {"Authorization": f"KakaoAK {key}"}

    # Nearest notable place within 100m. Categories: tourist attraction,
    # cultural facility, cafe, restaurant, accommodation, subway.
    for category in ("AT4", "CT1", "CE7", "FD6", "AD5", "SW8"):
        params = urllib.parse.urlencode(
            {"category_group_code": category, "x": lng, "y": lat, "radius": 100, "size": 1, "sort": "distance"}
        )
        data = _get_json(f"{KAKAO_CATEGORY_SEARCH}?{params}", headers)
        documents = (data or {}).get("documents") or []
        if documents:
            return documents[0].get("place_name")

    params = urllib.parse.urlencode({"x": lng, "y": lat})
    data = _get_json(f"{KAKAO_COORD2ADDRESS}?{params}", headers)
    documents = (data or {}).get("documents") or []
    if not documents:
        return None
    entry = documents[0]
    road = entry.get("road_address") or {}
    address = entry.get("address") or {}
    return road.get("address_name") or address.get("address_name")


def yahoo_japan(lat: float, lng: float) -> Optional[str]:
    """Yahoo! JAPAN YOLP reverse geocoder."""
    client_id = os.environ.get("YAHOO_JP_CLIENT_ID")
    if not client_id or not _in_bbox(lat, lng, JAPAN_BBOX):
        return None
    params = urllib.parse.urlencode(
        {"lat": lat, "lon": lng, "appid": client_id, "output": "json"}
    )
    data = _get_json(f"{YAHOO_JP_REVERSE}?{params}")
    features = (data or {}).get("Feature") or []
    if not features:
        return None
    return (features[0].get("Property") or {}).get("Address") or features[0].get("Name")


def nominatim(lat: float, lng: float) -> Optional[str]:
    """OpenStreetMap. No key, works anywhere, but returns administrative
    addresses rather than venue names."""
    params = urllib.parse.urlencode({"lat": lat, "lon": lng, "format": "jsonv2", "zoom": 16})
    data = _get_json(f"{NOMINATIM_ENDPOINT}?{params}")
    return (data or {}).get("display_name")


PROVIDERS: tuple[Callable[[float, float], Optional[str]], ...] = (kakao, yahoo_japan, nominatim)


def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """Returns a human-readable place-name guess, or None if no provider
    has one. This is a *hint* shown alongside an unlabeled LocationCluster
    (§3 step 6) — not the user-assigned name on LocationCluster.name."""
    for provider in PROVIDERS:
        name = provider(lat, lng)
        if name:
            return name
    return None


# ---------------------------------------------------------------------
# Forward geocoding — the reverse direction: a typed place name/address to
# coordinates. Added for §4.4 photo-location editing, where a user often
# knows the *name* of where a photo was taken ("코롬방제과점") and shouldn't
# have to go find its coordinates first just to correct a photo's location.
# Same provider set and same declining-fallthrough pattern as reverse
# geocoding above — Kakao and Yahoo! JAPAN are regional and simply return
# nothing outside their country (a typed name isn't associated with
# coordinates yet, so there's no bounding box to pre-filter on the way
# reverse geocoding does; each provider just tries and declines on its own).
# ---------------------------------------------------------------------


def kakao_forward(name: str) -> Optional[tuple[float, float]]:
    """Kakao Local keyword search — Korean places/addresses only; returns
    None outside Korea rather than a wrong guess."""
    key = os.environ.get("KAKAO_REST_API_KEY")
    if not key:
        return None
    headers = {"Authorization": f"KakaoAK {key}"}
    params = urllib.parse.urlencode({"query": name, "size": 1})
    data = _get_json(f"{KAKAO_KEYWORD_SEARCH}?{params}", headers)
    documents = (data or {}).get("documents") or []
    if not documents:
        return None
    try:
        return float(documents[0]["y"]), float(documents[0]["x"])
    except (KeyError, TypeError, ValueError):
        return None


def yahoo_japan_forward(name: str) -> Optional[tuple[float, float]]:
    """Yahoo! JAPAN YOLP geocoder — Japanese addresses/places."""
    client_id = os.environ.get("YAHOO_JP_CLIENT_ID")
    if not client_id:
        return None
    params = urllib.parse.urlencode({"query": name, "appid": client_id, "output": "json"})
    data = _get_json(f"{YAHOO_JP_GEOCODER}?{params}")
    features = (data or {}).get("Feature") or []
    if not features:
        return None
    coords = ((features[0].get("Geometry") or {}).get("Coordinates") or "")
    # YOLP returns "lon,lat" (note the order — opposite of how this
    # module's functions all return their result).
    parts = coords.split(",")
    if len(parts) != 2:
        return None
    try:
        lng, lat = float(parts[0]), float(parts[1])
        return lat, lng
    except ValueError:
        return None


def nominatim_forward(name: str) -> Optional[tuple[float, float]]:
    """OpenStreetMap search — worldwide, no key, the catch-all fallback."""
    params = urllib.parse.urlencode({"q": name, "format": "jsonv2", "limit": 1})
    data = _get_json(f"{NOMINATIM_SEARCH_ENDPOINT}?{params}")
    if not data:  # a list, possibly empty — _get_json returns None on error
        return None
    try:
        entry = data[0]
        return float(entry["lat"]), float(entry["lon"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


FORWARD_PROVIDERS: tuple[Callable[[str], Optional[tuple[float, float]]], ...] = (
    kakao_forward,
    yahoo_japan_forward,
    nominatim_forward,
)


def forward_geocode(name: str) -> Optional[tuple[float, float]]:
    """Returns (lat, lng) for a typed place name/address, or None if no
    provider recognizes it. A miss is reported to the user as "couldn't
    find that" rather than silently doing nothing (unlike the reverse
    direction's hints, this result is about to be saved directly as the
    photo's location, so the user needs to know when it didn't work)."""
    name = name.strip()
    if not name:
        return None
    for provider in FORWARD_PROVIDERS:
        coords = provider(name)
        if coords:
            return coords
    return None


def configured_providers() -> list[str]:
    """Which providers can actually answer right now — for surfacing setup
    state rather than silently degrading to Nominatim everywhere."""
    available = ["nominatim"]
    if os.environ.get("KAKAO_REST_API_KEY"):
        available.insert(0, "kakao")
    if os.environ.get("YAHOO_JP_CLIENT_ID"):
        available.insert(-1, "yahoo_japan")
    return available

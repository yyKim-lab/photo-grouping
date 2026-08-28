"""Local web UI for browsing and labeling clusters (§4.1-§4.3).

Runs on the user's own machine against their own SQLite database — not a
hosted service. Server-rendered HTML with a little vanilla JS: the labeling
flow is "look at a face, answer one question", which needs no build step,
no framework, and no client-side state.

Binds to 127.0.0.1 by default so a local install isn't inadvertently
serving someone's photo library to their whole network.

Start with:
    .venv/bin/python -m photo_grouping.web
"""

from __future__ import annotations

import functools
import io
import json
import os
import threading
import urllib.error
import uuid
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import sqlite3
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_file, url_for

from . import (
    autobio,
    db,
    export,
    face_embeddings,
    geocoding,
    google_auth,
    i18n,
    ingestion,
    llm,
    ocr,
    picker_client,
    repository,
    seed_import,
    storage,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "photo_grouping.db"
_REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SECRET_PATH = _REPO_ROOT / "secrets" / "client_secret.json"
TOKEN_CACHE_PATH = _REPO_ROOT / "secrets" / "token.json"
# Same paths llm.py's own resolve_api_key()/resolve_openai_key() read from
# by default — the Settings-page save routes below just write to these
# instead of requiring a self-hoster to create the files by hand.
ANTHROPIC_KEY_PATH = _REPO_ROOT / "secrets" / "anthropic_api_key.txt"
OPENAI_KEY_PATH = _REPO_ROOT / "secrets" / "openai_api_key.txt"
DEFAULT_ORIGINALS_DIR = _REPO_ROOT / "data" / "originals"
# Footer link (base.html) — empty until the project has a public repo to
# point to; set once it exists rather than linking to a 404.
GITHUB_URL = "https://github.com/yyKim-lab/photo-grouping"

app = Flask(__name__)
app.config["DB_PATH"] = DEFAULT_DB_PATH
# Face crops are re-cut from originals on every request. At personal scale
# that is fast enough and avoids a cache to invalidate when a user's edits
# move faces between clusters.
app.config["THUMBNAIL_MAX_PX"] = 400
app.config["FULLSCREEN_MAX_PX"] = 1800
# Local-device import (see /import/local below) reads whole files into
# memory; a generous but finite cap keeps an accidental huge-folder upload
# from taking the process down instead of just failing that request.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB per request


def get_conn() -> sqlite3.Connection:
    if "conn" not in g:
        g.conn = db.connect(app.config["DB_PATH"])
    return g.conn


def _complete_fn(conn: sqlite3.Connection):
    """The `complete` callable to hand Autobio: llm.complete itself,
    unchanged, unless the Settings page's "AI provider" choice pins a
    specific one — see repository.set_llm_provider / llm.complete's
    `provider` kwarg. Left as plain llm.complete when unset (the common
    case) rather than always wrapping it, so every existing call site and
    test that expects exactly llm.complete's default auto-detect
    behavior keeps working unchanged."""
    provider = repository.get_app_settings(conn)["llm_provider"]
    if not provider:
        return llm.complete
    return functools.partial(llm.complete, provider=provider)


@app.teardown_appcontext
def close_conn(_exception):
    conn = g.pop("conn", None)
    if conn is not None:
        conn.close()


@app.context_processor
def inject_i18n():
    # `t` is bound to the current request's ui_language so every template
    # can call {{ t('some.key') }} without threading the language through
    # each render_template() call by hand. Cheap: get_conn() is cached on
    # `g` for the life of the request, so this doesn't add an extra
    # connection or query beyond the first settings lookup.
    conn = get_conn()
    lang = repository.get_app_settings(conn)["ui_language"]
    return {
        "t": lambda key, **kw: i18n.t(key, lang, **kw),
        "ui_language": lang,
        "nav_active": _NAV_SECTIONS.get(request.endpoint),
        "github_url": GITHUB_URL,
    }


# Which nav-bar item is "current" for a given page — the CSS support
# (header a.active) existed from the start but nothing ever set the
# class, so the nav never actually showed where you were. Only page-view
# (GET, template-rendering) endpoints are listed: POST endpoints redirect
# immediately, so a user never really "sees" the nav mid-POST — there's
# nothing to highlight for those. A few page views genuinely don't belong
# to any one tab (e.g. seed_import_form, review_duplicates) and are left
# out on purpose, same as an unrecognized endpoint: nav_active is None,
# and no link gets the active class.
_NAV_SECTIONS = {
    "index": "albums",
    "cluster_detail": "albums",
    "excluded": "albums",
    "timeline": "photos",
    "photo_detail": "photos",
    "events_index": "groups",
    "event_detail": "groups",
    "queue": "queue",
    "autobio_daily_index": "diary",
    "autobio_entry": "diary",
    "autobio_index": "autobio",
    "autobio_summary_view": "autobio",
    "import_start_page": "import",
    "import_local_form": "import",
    "settings_page": "settings",
    "settings_connect_page": "settings",
}


# ---------------------------------------------------------------------
# Browsing (§4.1)
# ---------------------------------------------------------------------


@app.route("/")
def index():
    conn = get_conn()
    return render_template(
        "index.html",
        albums=repository.load_albums(conn),
        uncategorized=repository.load_uncategorized(conn),
        queue_length=len(repository.load_labeling_queue(conn)),
        duplicates=repository.duplicate_named_clusters(conn),
        excluded_count=repository.excluded_count(conn),
        counts=repository.counts(conn),
    )


# ---------------------------------------------------------------------
# Labeling queue (§4.2)
# ---------------------------------------------------------------------


@app.route("/queue")
def queue():
    conn = get_conn()
    items = repository.load_labeling_queue(conn)
    if not items:
        return render_template(
            "queue_empty.html",
            merged=_merged_notice(),
            undetected_photos=repository.photos_with_nothing_detected(conn),
        )

    item = items[0]
    context = {
        "item": item,
        "remaining": len(items),
        "merged": _merged_notice(),
        "speech_language": repository.get_app_settings(conn)["speech_language"],
    }
    if item["kind"] == "face":
        context["face"] = repository.representative_face(conn, item["id"])
    else:
        context["photos"] = repository.photos_in_location_cluster(conn, item["id"])[:6]
        context["suggestions"] = _name_suggestions(
            repository.cluster_row(conn, "place", item["id"])
        )
    return render_template("queue.html", **context)


def _name_suggestions(row) -> list[dict]:
    """Name suggestions for a place, most trustworthy first: OCR text read
    off the photo, then the map provider's answer.

    OCR leads because a storefront sign names a venue where a map often only
    knows the street — but both are offered, because OCR misreads freely
    (see ocr.py). Neither is ever applied automatically."""
    if row is None:
        return []
    suggestions = [
        {"text": c["text"], "source": "sign", "confidence": c.get("confidence")}
        for c in ocr.decode_candidates(row["ocr_name"])
    ]
    if row["geocoded_name"]:
        # Map answers are often a full address; the leading component is
        # usually the useful part ("코롬방제과점, 무안동, 목포시, ...").
        first = row["geocoded_name"].split(",")[0].strip()
        suggestions.append({"text": first, "source": "map", "confidence": None})
        if first != row["geocoded_name"]:
            suggestions.append(
                {"text": row["geocoded_name"], "source": "map (full)", "confidence": None}
            )
    return suggestions


@app.post("/cluster/<kind>/<int:cluster_id>/name")
def name_cluster(kind, cluster_id):
    name = (request.form.get("name") or "").strip()
    if not name:
        # An empty name would create a "named" cluster with no label, which
        # would vanish from both Albums and Uncategorized. Bounce back.
        return redirect(request.referrer or url_for("queue"))
    conn = get_conn()
    with conn:
        repository.name_cluster(conn, kind, cluster_id, name)

    # If this name is now on more than one cluster, the user has just told
    # us those clusters are the same person — a stronger signal than any
    # similarity score. Ask right here, while they're still thinking about
    # this person, rather than leaving it to be noticed later.
    if len(repository.clusters_sharing_a_name(conn, kind, name)) > 1:
        return redirect(url_for("same_name", kind=kind, name=name))

    return redirect(request.form.get("next") or url_for("queue"))


@app.post("/cluster/<kind>/<int:cluster_id>/description")
def set_cluster_description(kind, cluster_id):
    conn = get_conn()
    try:
        with conn:
            repository.set_cluster_description(conn, kind, cluster_id, request.form.get("description") or "")
    except ValueError:
        abort(404)
    return redirect(url_for("cluster_detail", kind=kind, cluster_id=cluster_id))


@app.route("/same-name/<kind>/<name>")
def same_name(kind, name):
    conn = get_conn()
    clusters = repository.clusters_sharing_a_name(conn, kind, name)
    if len(clusters) < 2:
        return redirect(url_for("index"))
    return render_template(
        "same_name.html",
        kind=kind,
        name=name,
        clusters=clusters,
        # Whatever the user was doing before this prompt interrupted them,
        # so merging returns them to it rather than into the album.
        next_url=request.args.get("next") or url_for("queue"),
    )


@app.post("/same-name/<kind>/<name>/merge")
def merge_same_name(kind, name):
    conn = get_conn()
    ids = [c["id"] for c in repository.clusters_sharing_a_name(conn, kind, name)]
    if len(ids) < 2:
        return redirect(url_for("index"))
    with conn:
        if kind == "face":
            survivor = repository.merge_face_clusters(conn, ids)
        else:
            survivor = repository.merge_location_clusters(conn, ids)

    # Return to where the merge was started from — working through a list of
    # duplicates shouldn't be interrupted by a jump into one album. The
    # merged album is offered as a link instead of being forced.
    destination = request.form.get("next") or url_for("queue")
    return redirect(
        f"{destination}?{urlencode({'merged_kind': kind, 'merged_id': survivor, 'merged_name': name})}"
    )


@app.route("/review-duplicates")
def review_duplicates():
    return render_template(
        "review_duplicates.html",
        duplicates=repository.duplicate_named_clusters(get_conn()),
        merged=_merged_notice(),
    )


def _merged_notice() -> Optional[dict]:
    """Details of a merge that just happened, passed through the redirect as
    query params so a confirmation can be shown without needing Flask
    sessions (and therefore without a secret key to configure)."""
    if not request.args.get("merged_id"):
        return None
    return {
        "kind": request.args.get("merged_kind", "face"),
        "id": request.args.get("merged_id"),
        "name": request.args.get("merged_name", ""),
    }


@app.post("/cluster/<kind>/<int:cluster_id>/exclude")
def exclude_cluster(kind, cluster_id):
    """Hide a cluster — a stranger in the background, a face on a poster.
    Nothing is deleted; it just stops being offered, and can be restored."""
    conn = get_conn()
    try:
        with conn:
            repository.exclude_cluster(conn, kind, cluster_id)
    except ValueError:
        abort(404)
    return redirect(request.form.get("next") or url_for("queue"))


@app.post("/cluster/<kind>/<int:cluster_id>/restore")
def restore_cluster(kind, cluster_id):
    conn = get_conn()
    try:
        with conn:
            repository.restore_cluster(conn, kind, cluster_id)
    except ValueError:
        abort(404)
    return redirect(request.form.get("next") or url_for("excluded"))


@app.route("/excluded")
def excluded():
    conn = get_conn()
    return render_template("excluded.html", excluded=repository.load_excluded(conn))


@app.post("/cluster/<kind>/<int:cluster_id>/not-sure")
def not_sure(kind, cluster_id):
    conn = get_conn()
    with conn:
        repository.mark_cluster_not_sure(conn, kind, cluster_id)
    return redirect(request.form.get("next") or url_for("queue"))


# ---------------------------------------------------------------------
# Cluster detail (§4.3)
# ---------------------------------------------------------------------


@app.route("/cluster/<kind>/<int:cluster_id>")
def cluster_detail(kind, cluster_id):
    conn = get_conn()
    try:
        row = repository.cluster_row(conn, kind, cluster_id)
    except ValueError:
        abort(404)
    if row is None:
        abort(404)

    context = {"kind": kind, "cluster": dict(row), "suggestions": []}
    if kind == "place":
        context["suggestions"] = _name_suggestions(row)
        context["photos"] = repository.photos_in_location_cluster(conn, cluster_id)
        context["candidates"] = []
    else:
        # Two views of the same person: "photos" for browsing (the actual
        # pictures — what §4.1 means by an album), "faces" for split (cropped
        # faces are for judging "is this the same person?", not for looking
        # at your own photos). Photos is the default; faces is reached via
        # an explicit toggle since split needs face_instance ids the photo
        # view doesn't carry.
        view = request.args.get("view", "photos")
        context["view"] = view if view in ("photos", "faces") else "photos"
        context["photos"] = repository.photos_in_face_cluster(conn, cluster_id)
        context["faces"] = repository.face_instances_in_cluster(conn, cluster_id)
        # Ranked merge candidates (§4.3). The clustering threshold is
        # deliberately conservative, so one person split in two is expected
        # — the other half is usually the top candidate here.
        context["candidates"] = repository.similar_face_clusters(conn, cluster_id)
    return render_template("cluster_detail.html", **context)


@app.post("/cluster/<kind>/<int:cluster_id>/merge")
def merge_clusters(kind, cluster_id):
    other_ids = [int(v) for v in request.form.getlist("other_id")]
    if not other_ids:
        return redirect(url_for("cluster_detail", kind=kind, cluster_id=cluster_id))

    conn = get_conn()
    ids = sorted({cluster_id, *other_ids})
    try:
        with conn:
            if kind == "face":
                survivor = repository.merge_face_clusters(conn, ids)
            elif kind == "place":
                survivor = repository.merge_location_clusters(conn, ids)
            else:
                abort(404)
    except ValueError:
        abort(400)
    return redirect(url_for("cluster_detail", kind=kind, cluster_id=survivor))


@app.post("/cluster/face/<int:cluster_id>/split")
def split_cluster(cluster_id):
    face_ids = [int(v) for v in request.form.getlist("face_instance_id")]
    if not face_ids:
        return redirect(url_for("cluster_detail", kind="face", cluster_id=cluster_id))

    conn = get_conn()
    try:
        with conn:
            new_id = repository.split_face_cluster(conn, cluster_id, face_ids)
    except ValueError as e:
        return render_template("error.html", message=str(e)), 400
    return redirect(url_for("cluster_detail", kind="face", cluster_id=new_id))


# ---------------------------------------------------------------------
# Per-photo metadata edits (§4.4)
# ---------------------------------------------------------------------


@app.route("/photo/<int:photo_id>")
def photo_detail(photo_id):
    conn = get_conn()
    detail = repository.photo_detail(conn, photo_id)
    if detail is None:
        abort(404)
    photo = detail["photo"]
    return render_template(
        "photo_detail.html",
        photo_id=photo_id,
        photo=photo,
        faces=detail["faces"],
        location=detail["location"],
        effective_taken_at=photo["taken_at_override"] or photo["taken_at"],
        speech_language=repository.get_app_settings(conn)["speech_language"],
    )


@app.post("/photo/<int:photo_id>/face/<int:face_instance_id>/reassign")
def reassign_face(photo_id, face_instance_id):
    name = (request.form.get("name") or "").strip()
    if not name:
        return redirect(url_for("photo_detail", photo_id=photo_id))
    conn = get_conn()
    with conn:
        repository.reassign_face(conn, face_instance_id, name)
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/face/<int:face_instance_id>/false-positive")
def mark_false_positive(photo_id, face_instance_id):
    conn = get_conn()
    with conn:
        repository.mark_face_false_positive(conn, face_instance_id)
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/add-face")
def add_face(photo_id):
    conn = get_conn()
    name = (request.form.get("name") or "").strip()
    try:
        bounding_box = {
            "x": float(request.form["x"]),
            "y": float(request.form["y"]),
            "width": float(request.form["width"]),
            "height": float(request.form["height"]),
        }
    except (KeyError, ValueError):
        return render_template("error.html", message="Invalid selection — try drawing the box again."), 400
    if not name:
        return render_template("error.html", message="A name is needed to add a face."), 400

    row = conn.execute("SELECT original_storage_path FROM photo WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        abort(404)
    path = Path(row["original_storage_path"])
    if not path.exists():
        return render_template("error.html", message="Original photo file not found on disk."), 404

    result = face_embeddings.embed_manual_crop(path.read_bytes(), bounding_box)
    if result is None:
        return render_template(
            "error.html",
            message="No face found in that selection — try drawing a closer box around just the face.",
        ), 400
    refined_box, embedding = result

    with conn:
        repository.add_manual_face(
            conn, photo_id=photo_id, bounding_box=refined_box, embedding=embedding, name=name
        )
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/date")
def set_photo_date(photo_id):
    from datetime import datetime

    raw = (request.form.get("taken_at") or "").strip()
    try:
        # <input type="datetime-local"> posts "YYYY-MM-DDTHH:MM".
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return render_template("error.html", message="Could not understand that date/time."), 400

    conn = get_conn()
    with conn:
        repository.override_taken_at(conn, photo_id, parsed.isoformat())
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/location")
def set_photo_location_override(photo_id):
    try:
        lat = float(request.form["lat"])
        lng = float(request.form["lng"])
    except (KeyError, ValueError):
        return render_template("error.html", message="Latitude/longitude must be numbers."), 400
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return render_template("error.html", message="That's outside valid latitude/longitude range."), 400

    conn = get_conn()
    with conn:
        repository.override_photo_location(conn, photo_id, lat, lng)
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/location-by-name")
def set_photo_location_by_name(photo_id):
    # A person usually knows a photo's location by *name*, not by
    # coordinates — this looks the name up (reusing the same regional
    # provider chain §3 uses for reverse geocoding, just run forward) and
    # saves it through the exact same override path as the lat/lng form
    # above, so both routes end up with identical, consistent behavior.
    name = (request.form.get("place_name") or "").strip()
    if not name:
        return render_template("error.html", message="Type a place name to look up."), 400

    coords = geocoding.forward_geocode(name)
    if coords is None:
        return render_template(
            "error.html",
            message=f"Couldn't find a location for “{name}”. Try different wording, "
                     "or enter coordinates directly instead.",
        ), 400

    conn = get_conn()
    with conn:
        repository.override_photo_location(conn, photo_id, coords[0], coords[1])
    return redirect(url_for("photo_detail", photo_id=photo_id))


@app.post("/photo/<int:photo_id>/description")
def set_photo_description(photo_id):
    conn = get_conn()
    with conn:
        repository.set_photo_description(conn, photo_id, request.form.get("description") or "")
    return redirect(url_for("photo_detail", photo_id=photo_id))


# ---------------------------------------------------------------------
# §4.6b — download a photo's original on request. First cut of this
# wired it straight to a fresh Google Picker re-pick every time, following
# §4.6b's own text literally — real use immediately showed that's a bad
# experience ("makes me search the photo again"), and it turns out to
# also be unnecessary: §5's ingestion design already fetches and
# permanently retains a full-quality original locally for *every* photo
# (see ingestion.py's storage_adapter.save_original() call, using the
# same '=d' full-quality fetch) — §4.6b's Google-re-pick text reads like
# it predates that §5 decision and was never reconciled with it. So the
# common case is now just: serve the file already sitting in
# data/originals/, instantly, no Google interaction at all. The Picker
# re-pick flow is kept as a fallback for the one case it's genuinely
# needed: the local file has gone missing (moved, deleted, or imported
# before this app existed) — not the primary path anymore.
# ---------------------------------------------------------------------

_pending_downloads: dict[int, dict] = {}
_pending_downloads_lock = threading.Lock()


@app.route("/photo/<int:photo_id>/download-original")
def download_original_start(photo_id):
    conn = get_conn()
    detail = repository.photo_detail(conn, photo_id)
    if detail is None:
        abort(404)
    photo = detail["photo"]

    local_path = Path(photo["original_storage_path"]) if photo["original_storage_path"] else None
    if local_path and local_path.exists():
        # The common case: already have it, just send it — no page, no
        # Google, no waiting.
        return send_file(
            local_path, as_attachment=True, download_name=photo["original_filename"] or local_path.name
        )

    return render_template(
        "download_original_start.html",
        photo_id=photo_id,
        photo=photo,
        client_secret_missing=not CLIENT_SECRET_PATH.exists(),
    )


@app.post("/photo/<int:photo_id>/download-original/start")
def download_original_session_start(photo_id):
    if not CLIENT_SECRET_PATH.exists():
        return render_template(
            "error.html",
            message=f"Missing {CLIENT_SECRET_PATH} — see README.md 'Google OAuth setup'.",
        ), 400

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)
    session = picker_client.create_session(access_token)

    with _pending_downloads_lock:
        _pending_downloads[photo_id] = {"session_id": session["id"], "picker_uri": session["pickerUri"]}

    return render_template(
        "download_original_picking.html",
        photo_id=photo_id,
        session_id=session["id"],
        picker_uri=session["pickerUri"],
    )


@app.get("/photo/<int:photo_id>/download-original/status")
def download_original_status(photo_id):
    session_id = request.args.get("session_id")
    if not session_id:
        abort(400)
    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)
    try:
        session = picker_client.get_session(access_token, session_id)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return jsonify({"ready": False, "expired": True})
        raise
    return jsonify({"ready": bool(session.get("mediaItemsSet"))})


@app.post("/photo/<int:photo_id>/download-original/fetch")
def download_original_fetch(photo_id):
    session_id = request.form.get("session_id")
    picker_uri = request.form.get("picker_uri")
    if not session_id:
        abort(400)

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)

    try:
        session = picker_client.get_session(access_token, session_id)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            with _pending_downloads_lock:
                _pending_downloads.pop(photo_id, None)
            return render_template(
                "error.html", message="This picking session expired. Start over."
            ), 410
        raise

    if not session.get("mediaItemsSet"):
        return render_template(
            "download_original_picking.html",
            photo_id=photo_id,
            session_id=session_id,
            picker_uri=picker_uri,
            not_ready=True,
        )

    items = picker_client.list_media_items(access_token, session_id)
    if not items:
        return render_template("error.html", message="No photo was selected."), 400
    # Only one photo should ever be selected here — the picking page asks
    # for exactly this one — but take the first regardless, rather than
    # erroring on extras picked by mistake.
    item = items[0]
    original_bytes = picker_client.fetch_original_bytes(access_token, item["mediaFile"]["baseUrl"])
    filename = item["mediaFile"].get("filename") or f"photo-{photo_id}.jpg"

    picker_client.delete_session(access_token, session_id)
    with _pending_downloads_lock:
        _pending_downloads.pop(photo_id, None)

    return send_file(
        io.BytesIO(original_bytes), mimetype="image/jpeg", as_attachment=True, download_name=filename
    )


# ---------------------------------------------------------------------
# Bulk seed import (§4.5)
# ---------------------------------------------------------------------


@app.route("/seed-import")
def seed_import_form():
    return render_template("seed_import_form.html")


@app.post("/seed-import/detect")
def seed_import_detect():
    import base64

    upload = request.files.get("screenshot")
    if not upload or not upload.filename:
        return render_template("error.html", message="Choose a screenshot to upload first."), 400

    image_bytes = upload.read()
    try:
        candidates = seed_import.detect_seed_candidates(image_bytes)
    except Exception as e:  # noqa: BLE001 - a corrupt/unreadable upload shouldn't 500
        return render_template("error.html", message=f"Could not read that image: {e}"), 400

    if not candidates:
        return render_template(
            "error.html", message="No faces found in that screenshot."
        ), 400

    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    rows = []
    for candidate in candidates:
        box = candidate["bounding_box"]
        left, top = box["x"] * width, box["y"] * height
        right, bottom = left + box["width"] * width, top + box["height"] * height
        crop = image.crop((int(left), int(top), int(right), int(bottom)))
        crop.thumbnail((200, 200))
        buf = BytesIO()
        crop.save(buf, "JPEG", quality=85)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        rows.append(
            {
                "crop_data_uri": data_uri,
                "guessed_name": candidate["guessed_name"],
                "embedding_b64": base64.b64encode(
                    repository.encode_embedding(candidate["embedding"])
                ).decode(),
            }
        )

    return render_template("seed_import_confirm.html", rows=rows)


@app.post("/seed-import/save")
def seed_import_save():
    import base64

    try:
        count = int(request.form["count"])
    except (KeyError, ValueError):
        abort(400)

    conn = get_conn()
    saved = 0
    with conn:
        for i in range(count):
            if not request.form.get(f"keep_{i}"):
                continue
            name = (request.form.get(f"name_{i}") or "").strip()
            embedding_b64 = request.form.get(f"embedding_{i}")
            if not name or not embedding_b64:
                continue
            embedding = repository.decode_embedding(base64.b64decode(embedding_b64))
            repository.insert_seed_face(conn, name=name, embedding=embedding)
            saved += 1

    return render_template("seed_import_done.html", saved=saved)


# ---------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------


def _open_photo(conn, photo_id: int):
    from PIL import Image

    row = conn.execute(
        "SELECT original_storage_path FROM photo WHERE id = ?", (photo_id,)
    ).fetchone()
    if row is None:
        abort(404)
    path = Path(row["original_storage_path"])
    if not path.exists():
        # The original may live on a cloud backend, or have been moved.
        abort(404, "Original file not found on disk.")
    return Image.open(path)


def _send(image, fmt="JPEG", quality=85):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, fmt, quality=quality)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")


@app.route("/photo/<int:photo_id>/thumb")
def photo_thumb(photo_id):
    image = _open_photo(get_conn(), photo_id)
    image.thumbnail((app.config["THUMBNAIL_MAX_PX"], app.config["THUMBNAIL_MAX_PX"]))
    return _send(image)


@app.route("/photo/<int:photo_id>/full")
def photo_full(photo_id):
    """A large rendition for the full-screen viewer — bigger than the grid
    thumbnail, but still capped and re-encoded rather than serving the raw
    original: originals from a modern phone run 3-10MB+, and re-encoding at
    a viewport-sized cap keeps the lightbox responsive on a real library
    without the disk-space or bandwidth cost of the untouched file."""
    image = _open_photo(get_conn(), photo_id)
    image.thumbnail((app.config["FULLSCREEN_MAX_PX"], app.config["FULLSCREEN_MAX_PX"]))
    return _send(image, quality=90)


@app.route("/face/<int:face_instance_id>/crop")
def face_crop(face_instance_id):
    import json

    conn = get_conn()
    row = conn.execute(
        "SELECT photo_id, bounding_box FROM face_instance WHERE id = ?", (face_instance_id,)
    ).fetchone()
    if row is None:
        abort(404)

    image = _open_photo(conn, row["photo_id"])
    box = json.loads(row["bounding_box"])
    width, height = image.size

    # Bounding boxes are stored as 0..1 fractions, so they scale to
    # whatever resolution the original happens to be.
    left, top = box["x"] * width, box["y"] * height
    right, bottom = left + box["width"] * width, top + box["height"] * height

    # A little context around the face reads much better than a tight crop —
    # hair and chin are most of what makes someone recognisable at a glance.
    pad_x, pad_y = box["width"] * width * 0.4, box["height"] * height * 0.4
    crop = image.crop(
        (
            max(0, int(left - pad_x)),
            max(0, int(top - pad_y)),
            min(width, int(right + pad_x)),
            min(height, int(bottom + pad_y)),
        )
    )
    crop.thumbnail((app.config["THUMBNAIL_MAX_PX"], app.config["THUMBNAIL_MAX_PX"]))
    return _send(crop)


@app.route("/cluster/face/<int:cluster_id>/crop")
def face_crop_for_cluster(cluster_id):
    """A cluster's cover image — its representative face."""
    face = repository.representative_face(get_conn(), cluster_id)
    if face is None:
        abort(404)
    return face_crop(face["face_instance_id"])


@app.route("/cluster/place/<int:cluster_id>/thumb")
def place_thumb(cluster_id):
    """A place cluster's cover image — its first photo, since a location
    has no equivalent of a representative face."""
    photos = repository.photos_in_location_cluster(get_conn(), cluster_id)
    if not photos:
        abort(404)
    return photo_thumb(photos[0]["photo_id"])


# ---------------------------------------------------------------------
# Events — new, not in the original spec. A purely user-created grouping:
# select photos, name the occasion. See repository.py's Events section for
# why this is its own table rather than reusing FaceCluster/LocationCluster.
# ---------------------------------------------------------------------


@app.route("/events")
def events_index():
    return render_template("events_index.html", events=repository.list_events(get_conn()))


@app.route("/event/<int:event_id>/cover")
def event_cover(event_id):
    """An event's cover image — its first photo, same idea as a place
    cluster's cover (there's no single 'representative' shot for an
    occasion the way there is a representative face for a person)."""
    detail = repository.event_detail(get_conn(), event_id)
    if not detail or not detail["photos"]:
        abort(404)
    return photo_thumb(detail["photos"][0]["photo_id"])


@app.route("/event/<int:event_id>")
def event_detail(event_id):
    conn = get_conn()
    detail = repository.event_detail(conn, event_id)
    if detail is None:
        abort(404)
    return render_template("event_detail.html", event=detail["event"], photos=detail["photos"])


@app.post("/event/<int:event_id>/rename")
def event_rename(event_id):
    name = (request.form.get("name") or "").strip()
    if not name:
        return redirect(url_for("event_detail", event_id=event_id))
    conn = get_conn()
    try:
        with conn:
            repository.rename_event(conn, event_id, name)
    except ValueError as e:
        return render_template("error.html", message=str(e)), 400
    return redirect(url_for("event_detail", event_id=event_id))


@app.post("/event/<int:event_id>/description")
def event_set_description(event_id):
    conn = get_conn()
    with conn:
        repository.set_event_description(conn, event_id, request.form.get("description") or "")
    return redirect(url_for("event_detail", event_id=event_id))


@app.post("/event/<int:event_id>/remove-photo")
def event_remove_photo(event_id):
    try:
        photo_id = int(request.form["photo_id"])
    except (KeyError, ValueError):
        abort(400)
    conn = get_conn()
    with conn:
        repository.remove_photo_from_event(conn, event_id, photo_id)
    return redirect(url_for("event_detail", event_id=event_id))


@app.post("/event/<int:event_id>/autobio-exclude")
def event_set_autobio_excluded(event_id):
    conn = get_conn()
    excluded = request.form.get("excluded") == "1"
    with conn:
        repository.set_event_autobio_excluded(conn, event_id, excluded)
    return redirect(url_for("event_detail", event_id=event_id))


@app.post("/event/<int:event_id>/delete")
def event_delete(event_id):
    conn = get_conn()
    with conn:
        repository.delete_event(conn, event_id)
    return redirect(url_for("events_index"))


@app.post("/events/bulk-add")
def events_bulk_add():
    """The event-assignment action from the timeline view (§ 'photo by
    time'): select photos, type an event name, done — same match-existing-
    or-create-new pattern used for naming people and places."""
    name = (request.form.get("event_name") or "").strip()
    photo_ids = [int(v) for v in request.form.getlist("photo_id")]
    next_url = request.form.get("next") or url_for("timeline")

    if not name or not photo_ids:
        return redirect(next_url)

    conn = get_conn()
    with conn:
        event_id = repository.get_or_create_event(conn, name)
        repository.add_photos_to_event(conn, event_id, photo_ids)
    return redirect(url_for("event_detail", event_id=event_id))


# ---------------------------------------------------------------------
# Timeline — "photo by time": every photo, newest first, independent of
# whether it has been sorted into any person/place/event yet. Also where
# bulk event-assignment happens (select photos here, not one at a time).
# ---------------------------------------------------------------------


def _format_date_group_label(day: date) -> str:
    # "Tue, Aug 18" — matches Google Photos' own timeline headers. Year is
    # only appended for older photos; omitting it for the current year is
    # what makes the common case (browsing recent imports) read cleanly.
    label = day.strftime("%a, %b ") + str(day.day)
    if day.year != date.today().year:
        label += f", {day.year}"
    return label


def _group_photos_by_date(photos: list[dict]) -> list[dict]:
    """Photos already arrive newest-first, so same-day photos are already
    adjacent — this only needs one linear pass, not a full groupby/sort.
    Each group also gets a place name, but only when every photo in that
    day agrees on one — a day that visibly spans two places shouldn't
    silently show just the first one."""
    groups: list[dict] = []
    current_day = None
    for photo in photos:
        taken_at = photo.get("taken_at") or ""
        try:
            dt = datetime.fromisoformat(taken_at.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        day = dt.date() if dt else None
        if day != current_day or not groups:
            current_day = day
            groups.append(
                {
                    "date_label": _format_date_group_label(day) if day else "Unknown date",
                    "place_names": set(),
                    "photos": [],
                }
            )
        groups[-1]["photos"].append(photo)
        if photo.get("place_name"):
            groups[-1]["place_names"].add(photo["place_name"])
    for group in groups:
        names = group.pop("place_names")
        group["place_name"] = next(iter(names)) if len(names) == 1 else None
    return groups


@app.route("/timeline")
def timeline():
    conn = get_conn()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 60
    photos = repository.list_all_photos(conn, limit=per_page, offset=(page - 1) * per_page)
    total = repository.total_photo_count(conn)
    return render_template(
        "timeline.html",
        groups=_group_photos_by_date(photos),
        page=page,
        total_pages=max(1, -(-total // per_page)),
        total=total,
        # Backs a <datalist> on the event-name field: typing shows matching
        # existing events to pick from, but an unmatched value still works
        # — events_bulk_add() below creates it via get_or_create_event().
        event_names=[e["name"] for e in repository.list_events(conn)],
    )


# ---------------------------------------------------------------------
# Import from Google Photos — new: this used to be CLI-only
# (scripts/ingest.py). Split across two requests rather than one, because
# the flow genuinely has two separate human waits in the middle of it: OAuth
# consent (once) and picking photos in Google's own UI (every time) — a
# single request would have to block through both.
#
# Fetching+ingesting the picked photos is itself a third wait — real
# batches take long enough that a synchronous POST left the browser on a
# blank page with no sign anything was happening. That work now runs in a
# background thread (its own sqlite connection — a bare thread has no
# Flask request context / no `g`) while the browser polls for progress,
# mirroring the /import/status polling pattern already used earlier in
# this flow. In-memory only (no DB table): a job's progress is only ever
# interesting to the one tab watching it, and doesn't need to survive a
# server restart the way the pending-session record does.
# ---------------------------------------------------------------------


class _ImportJob:
    def __init__(self, total: int, source: str):
        self.total = total
        self.source = source  # "google" | "local" — which done-page copy to show
        self.current = 0
        self.last = ""
        self.state = "running"  # running | done | error
        self.result: Optional[ingestion.IngestionResult] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def progress(self, filename: str, status: str) -> None:
        with self._lock:
            self.current += 1
            self.last = f"{filename}: {status}"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "current": self.current,
                "total": self.total,
                "last": self.last,
                "error": self.error,
            }


_import_jobs: dict[str, _ImportJob] = {}
_import_jobs_lock = threading.Lock()

# How long a pending session is trusted before treating it as dead without
# even asking Google. Observed in practice: a session resumed ~21h after
# creation still passed GET /v1/sessions/{id} cleanly (200, mediaItemsSet
# false, expireTime a week out) — the session *record* was fine — but its
# pickerUri opened to Google's own "Couldn't open Google Photos" error
# page. So the picker UI dies well before the session API says it has,
# and there's no documented field to check for that specifically. An hour
# is a conservative guess at "the user meant to come back soon, not
# tomorrow" — generous for a normal picking session, but short enough to
# stop offering a resume link that's already known to be broken.
_PICKER_UI_STALE_AFTER = timedelta(hours=1)


def _pending_session_is_stale(pending: dict) -> bool:
    created_at = datetime.fromisoformat(pending["created_at"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc) - created_at > _PICKER_UI_STALE_AFTER


@app.route("/import")
def import_start_page():
    conn = get_conn()
    pending = repository.get_pending_import_session(conn)
    if pending and _pending_session_is_stale(pending):
        # Don't offer a "Finish that import" link that's already known to
        # be dead (see _PICKER_UI_STALE_AFTER) — quietly drop it instead,
        # same end state as if the user had never started it.
        with conn:
            repository.clear_pending_import_session(conn)
        pending = None
    return render_template(
        "import_start.html",
        client_secret_missing=not _google_configured(),
        # Server-side, not tab-side: if the picker replaced the original
        # tab instead of opening a new one (confirmed in practice — "no
        # tab to switch back to" after finishing in Google's picker UI),
        # that tab's in-memory session_id is gone for good. This survives
        # that, since it's read fresh from the database on every visit to
        # this page, from any tab/window/browser.
        pending=pending,
    )


@app.route("/import/start", methods=["GET", "POST"])
def import_start():
    # Accepts GET (from the template's plain <a target="_blank"> link) as
    # well as POST (kept for existing callers/tests) — see import_start.html
    # for why a GET link, not a POST form, is what actually opens reliably
    # in a new tab across browsers.
    if not CLIENT_SECRET_PATH.exists():
        return render_template(
            "error.html",
            message=f"Missing {CLIENT_SECRET_PATH} — see README.md 'Google OAuth setup'.",
        ), 400

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    # Blocks for OAuth consent only the first time (or after a refresh
    # token dies) — opens a browser tab itself via google_auth's loopback
    # flow, same mechanism the CLI scripts already use.
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)
    session = picker_client.create_session(access_token)

    conn = get_conn()
    with conn:
        repository.save_pending_import_session(
            conn, session_id=session["id"], picker_uri=session["pickerUri"]
        )

    return render_template("import_picking.html", session_id=session["id"], picker_uri=session["pickerUri"])


@app.get("/import/status")
def import_status():
    # Polled by import_picking.html's JS so the page can auto-continue once
    # picking is done, instead of making the user remember to come back and
    # click a button — that manual step was the actual gap behind "I select
    # photos but nothing happens": nothing *was* supposed to happen until
    # the click, but that wasn't obvious from the picker tab alone.
    session_id = request.args.get("session_id")
    if not session_id:
        abort(400)
    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)
    try:
        session = picker_client.get_session(access_token, session_id)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # The session died server-side — nothing left to poll for.
            # Reported distinctly from "not ready yet" so the page can stop
            # polling instead of retrying a session that will never become
            # ready, rather than surfacing a raw 500 to the fetch() call.
            return jsonify({"ready": False, "expired": True})
        raise
    if not session.get("mediaItemsSet"):
        # Same "expired" signal for a session that's gone stale by our own
        # clock even though Google's API still reports it as live — see
        # _PICKER_UI_STALE_AFTER. A tab left open on this page past that
        # point would otherwise poll "not ready" forever against a picker
        # link that's already dead.
        conn = get_conn()
        pending = repository.get_pending_import_session(conn)
        if pending and pending["session_id"] == session_id and _pending_session_is_stale(pending):
            with conn:
                repository.clear_pending_import_session(conn)
            return jsonify({"ready": False, "expired": True})
    return jsonify({"ready": bool(session.get("mediaItemsSet"))})


@app.post("/import/continue")
def import_continue():
    session_id = request.form.get("session_id")
    # Carried through as a hidden field from import_picking.html rather than
    # re-read off get_session()'s response: per Google's documented fields
    # for GET /v1/sessions/{id}, only mediaItemsSet and pollingConfig are
    # guaranteed — pickerUri isn't, so reading it here risked a KeyError on
    # the exact "please finish picking and retry" path a user hits often.
    picker_uri = request.form.get("picker_uri")
    if not session_id:
        abort(400)

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)
    conn = get_conn()

    try:
        session = picker_client.get_session(access_token, session_id)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Commonly: too much time passed since /import/start and
            # Google expired the session server-side. Nothing to recover —
            # clear our record of it rather than offering a "resume" that
            # can only fail again, and say so plainly instead of a raw 500.
            with conn:
                repository.clear_pending_import_session(conn)
            return render_template(
                "error.html",
                message="This picking session has expired (they don't last forever). "
                         "Start a new import.",
            ), 410
        raise

    if not session.get("mediaItemsSet"):
        # Session record looks fine to Google's API even long after the
        # picker UI itself has stopped opening (see _PICKER_UI_STALE_AFTER)
        # — so "not ready yet" alone isn't enough to tell "still picking"
        # apart from "picker page is already dead and always will 404
        # against a fresh open." Only reachable via the /import hub's
        # resume banner (import_start_page already filters stale pendings
        # before offering it), but re-checked here too since this session
        # could just as easily have gone stale sitting on the auto-polling
        # picking page since it was opened.
        pending = repository.get_pending_import_session(conn)
        if pending and pending["session_id"] == session_id and _pending_session_is_stale(pending):
            with conn:
                repository.clear_pending_import_session(conn)
            return render_template(
                "error.html",
                message="This picking session has been open too long — Google's picker "
                         "page stops working well before the session record itself "
                         "expires. Start a new import.",
            ), 410
        # A single check, not a poll loop: the user clicked through to say
        # they're done, so if Google disagrees the likely explanation is
        # they're still picking — ask them to finish and try again, rather
        # than silently blocking the request for an unbounded time.
        return render_template(
            "import_picking.html",
            session_id=session_id,
            picker_uri=picker_uri,
            not_ready=True,
        )

    # Claim before doing any real work: the dedicated picking tab
    # (auto-polling) and the /import hub's resume banner are now two
    # independent live paths that can both land here for the very same
    # session — observed in practice as a real crash, where the loser's
    # list_media_items() 404s because the winner already fetched the items
    # and deleted the session out from under it. Only the winner proceeds;
    # a loser gets a calm "already handled" page instead of a stack trace.
    with conn:
        claimed = repository.claim_pending_import_session(conn, session_id)
    if not claimed:
        return render_template("import_already_handled.html")

    try:
        items = picker_client.list_media_items(access_token, session_id)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Claiming the pending-session row rules out the ordinary race
            # (two tabs both reaching this point), but this is one more
            # layer against the same underlying symptom from any other
            # cause — a stale/consumed session shouldn't surface as a raw
            # 500 either way.
            return render_template("import_already_handled.html")
        raise

    job = _ImportJob(total=len(items), source="google")
    with _import_jobs_lock:
        _import_jobs[session_id] = job
    db_path = app.config["DB_PATH"]

    def run() -> None:
        # Own connection: this runs on a bare thread with no Flask request
        # context, so there's no `g` to reuse get_conn() with.
        job_conn = db.connect(db_path)
        try:
            adapter = storage.get_adapter("local", _effective_originals_dir(job_conn))
            result = ingestion.ingest_picked_items(
                job_conn,
                items,
                access_token=access_token,
                picker_client=picker_client,
                storage_adapter=adapter,
                detect_faces=face_embeddings.detect_faces_in_bytes,
                on_progress=job.progress,
            )
            picker_client.delete_session(access_token, session_id)
            # Real bug hit in practice: this used to be the only ingestion
            # path, and scripts/ingest.py (the original, terminal-only way
            # to import) always called suggest_location_names() right after
            # — but that never got carried over when import moved into the
            # browser. Every web-imported photo's place clusters sat with
            # no geocoded/OCR suggestion forever, silently, since nothing
            # ever asked for one. Cheap when there's nothing new: it's a
            # no-op query over clusters actually missing a suggestion.
            ingestion.suggest_location_names(job_conn)
            job.result = result
            job.state = "done"
        except Exception as e:  # noqa: BLE001 - reported to the polling page, not raised in a thread with nowhere to go
            job.error = str(e)
            job.state = "error"
        finally:
            job_conn.close()

    threading.Thread(target=run, daemon=True).start()

    # The Google session_id doubles as the job id here (it's already a
    # unique per-import string) — local-device import (below) has no such
    # natural id, so it generates its own; import_progress.html just calls
    # it job_id either way and doesn't care which flow it came from.
    return render_template(
        "import_progress.html", job_id=session_id, picked_count=len(items)
    )


@app.get("/import/progress")
def import_progress():
    # Polled by import_progress.html while a background ingestion thread
    # (started in import_continue or import_local_upload) is running.
    job_id = request.args.get("job_id")
    if not job_id:
        abort(400)
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job.snapshot())


@app.get("/import/result")
def import_result():
    # Consumed once *the finished job* is fetched — a still-running job is
    # left in place (a premature/racing call here must not destroy a job
    # that hasn't finished yet; there'd be no way to ever retrieve it).
    job_id = request.args.get("job_id")
    if not job_id:
        abort(400)
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        if job is not None and job.state != "running":
            del _import_jobs[job_id]
    if job is None or job.state == "running":
        abort(404)
    if job.state == "error":
        return render_template(
            "error.html",
            message=f"The import ran into a problem partway through: {job.error}",
        ), 500
    return render_template(
        "import_done.html", result=job.result, picked_count=job.total, source=job.source
    )


@app.route("/import/local")
def import_local_form():
    return render_template("import_local_form.html")


@app.post("/import/local")
def import_local_upload():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return render_template("error.html", message="No files were selected."), 400

    # Read into plain bytes now, synchronously — these FileStorage objects
    # (and their backing temp files) belong to this request and won't
    # survive past it, so the background thread below needs its own copy
    # of the actual data, not a reference to the request-scoped upload.
    file_data = [(f.filename, f.read()) for f in files]

    job_id = uuid.uuid4().hex  # no natural id here, unlike a Google session_id
    job = _ImportJob(total=len(file_data), source="local")
    with _import_jobs_lock:
        _import_jobs[job_id] = job
    db_path = app.config["DB_PATH"]

    def run() -> None:
        job_conn = db.connect(db_path)
        try:
            adapter = storage.get_adapter("local", _effective_originals_dir(job_conn))
            result = ingestion.ingest_local_files(
                job_conn,
                file_data,
                storage_adapter=adapter,
                detect_faces=face_embeddings.detect_faces_in_bytes,
                on_progress=job.progress,
            )
            # See the matching comment in import_continue()'s run() above —
            # same gap, same fix, for the local-device import path.
            ingestion.suggest_location_names(job_conn)
            job.result = result
            job.state = "done"
        except Exception as e:  # noqa: BLE001 - reported to the polling page, not raised in a thread with nowhere to go
            job.error = str(e)
            job.state = "error"
        finally:
            job_conn.close()

    threading.Thread(target=run, daemon=True).start()

    return render_template(
        "import_progress.html", job_id=job_id, picked_count=len(file_data)
    )


# ---------------------------------------------------------------------
# Autobio (§4.6) — single-day entries plus a date-range combined
# narrative. Deferred: the per-segment tap-a-sentence photo-correction
# stepper, and a settings toggle for the unlabeled nudge (see autobio.py
# and README).
# ---------------------------------------------------------------------


@app.route("/autobio")
def autobio_index():
    # Combined narratives only — daily entries live on their own "Diary"
    # tab (autobio_daily_index) now, requested directly so the two don't
    # sit mixed together in one long list.
    conn = get_conn()
    summaries = repository.list_autobio_summaries(conn)
    for summary in summaries:
        summary["entry_dates"] = repository.autobio_entry_dates_for_ids(conn, summary["source_entry_ids"])
    return render_template("autobio_index.html", summaries=summaries, llm_configured=_llm_configured())


@app.route("/autobio/daily")
def autobio_daily_index():
    conn = get_conn()
    return render_template(
        "autobio_daily_index.html",
        entries=repository.list_autobio_entries(conn),
        settings=repository.get_autobio_settings(conn),
        llm_configured=_llm_configured(),
    )


@app.post("/autobio/settings")
def autobio_settings_save():
    conn = get_conn()
    with conn:
        repository.set_autobio_show_unlabeled_nudge(
            conn, bool(request.form.get("show_unlabeled_nudge"))
        )
    return redirect(url_for("autobio_daily_index"))


@app.post("/autobio/generate")
def autobio_generate():
    date = (request.form.get("date") or "").strip()
    if not date:
        return render_template("error.html", message="Pick a date first."), 400

    conn = get_conn()
    try:
        # complete=llm.complete passed explicitly (matching autobio's own
        # default) rather than left implicit: a Python default parameter
        # is bound once at import time, so patching llm.complete for a
        # test would silently miss the already-bound default — this forces
        # a fresh attribute lookup on every call instead.
        language = repository.get_app_settings(conn)["narrative_language"]
        autobio.generate_daily_entry(conn, date, complete=_complete_fn(conn), language=language)
    except autobio.NoPhotosForDate:
        return render_template(
            "error.html", message=f"No photos found for {date} — nothing to write about."
        ), 400
    except llm.LLMNotConfigured as e:
        return render_template("error.html", message=str(e)), 400
    except ValueError as e:
        # The model's response didn't parse into usable segments — a
        # genuine (if rare) LLM failure mode, not a bug to crash on.
        return render_template(
            "error.html", message=f"Couldn't generate a narrative for {date}: {e}"
        ), 502

    return redirect(url_for("autobio_entry", date=date))


@app.route("/autobio/<date>")
def autobio_entry(date):
    conn = get_conn()
    entry = repository.get_autobio_entry(conn, date)
    photo_count = len(repository.photos_for_date(conn, date))
    show_nudge = repository.get_autobio_settings(conn)["show_unlabeled_nudge"]
    # Computed live here rather than read off entry["has_unlabeled"]:
    # that stored flag is a snapshot from generation time, but labeling
    # can happen anytime afterward — the nudge should reflect right now,
    # not whatever was true when the entry was drafted. Skipped entirely
    # (not just hidden) when the setting is off, so a disabled nudge
    # doesn't still cost a query every page load.
    unlabeled_count = repository.count_unlabeled_for_date(conn, date) if (entry and show_nudge) else 0
    return render_template(
        "autobio_entry.html",
        date=date,
        entry=entry,
        photo_count=photo_count,
        unlabeled_count=unlabeled_count,
    )


@app.post("/autobio/<date>/save")
def autobio_save(date):
    conn = get_conn()
    with conn:
        repository.set_autobio_final_text(conn, date, request.form.get("final_text") or "")
    return redirect(url_for("autobio_entry", date=date))


@app.post("/autobio/<date>/segment/<int:index>/save")
def autobio_segment_save(date, index):
    conn = get_conn()
    with conn:
        try:
            repository.set_autobio_segment_text(conn, date, index, request.form.get("text") or "")
        except (ValueError, IndexError):
            abort(404)
    return redirect(url_for("autobio_entry", date=date))


@app.post("/autobio/<date>/segment/<int:index>/regenerate")
def autobio_segment_regenerate(date, index):
    conn = get_conn()
    try:
        language = repository.get_app_settings(conn)["narrative_language"]
        autobio.regenerate_segment(conn, date, index, complete=_complete_fn(conn), language=language)
    except (autobio.NoSuchSegment, autobio.NoPhotosForDate):
        abort(404)
    except llm.LLMNotConfigured as e:
        return render_template("error.html", message=str(e)), 400
    except ValueError as e:
        return render_template(
            "error.html", message=f"Couldn't regenerate that segment: {e}"
        ), 502
    return redirect(url_for("autobio_entry", date=date))


@app.post("/autobio/<date>/delete")
def autobio_entry_delete(date):
    conn = get_conn()
    with conn:
        repository.delete_autobio_entry(conn, date)
    return redirect(url_for("autobio_daily_index"))


@app.get("/autobio/<date>/export/<fmt>")
def autobio_export(date, fmt):
    conn = get_conn()
    entry = repository.get_autobio_entry(conn, date)
    if entry is None:
        abort(404)

    content = export.content_for_autobio_entry(entry)
    try:
        mimetype, data = export.export(content, fmt)
    except ValueError:
        abort(404)

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"autobio-{date}.{fmt}",
    )


# --- Combined narrative (§4.6 "Combined narrative") — a date-range
# summary built from daily entries, generating any missing ones first. ---


@app.post("/autobio/generate-range")
def autobio_generate_range():
    start_date = (request.form.get("start_date") or "").strip()
    end_date = (request.form.get("end_date") or "").strip()
    if not start_date or not end_date:
        return render_template("error.html", message="Pick both a start and end date."), 400

    conn = get_conn()
    try:
        language = repository.get_app_settings(conn)["narrative_language"]
        autobio.generate_combined_narrative(
            conn, start_date, end_date, complete=_complete_fn(conn), language=language
        )
    except autobio.NoEntriesForRange:
        return render_template(
            "error.html", message=f"No photos found between {start_date} and {end_date}."
        ), 400
    except llm.LLMNotConfigured as e:
        return render_template("error.html", message=str(e)), 400
    except ValueError as e:
        # Covers both "end date before start date" and a malformed
        # per-day LLM response bubbling up from generate_daily_entry.
        return render_template("error.html", message=str(e)), 400

    return redirect(url_for("autobio_summary_view", start_date=start_date, end_date=end_date))


@app.route("/autobio/summary/<start_date>/<end_date>")
def autobio_summary_view(start_date, end_date):
    conn = get_conn()
    summary = repository.get_autobio_summary(conn, start_date, end_date)
    return render_template(
        "autobio_summary.html", start_date=start_date, end_date=end_date, summary=summary
    )


@app.post("/autobio/summary/<start_date>/<end_date>/save")
def autobio_summary_save(start_date, end_date):
    conn = get_conn()
    with conn:
        repository.set_autobio_summary_text(conn, start_date, end_date, request.form.get("text") or "")
    return redirect(url_for("autobio_summary_view", start_date=start_date, end_date=end_date))


@app.post("/autobio/summary/<start_date>/<end_date>/delete")
def autobio_summary_delete(start_date, end_date):
    conn = get_conn()
    with conn:
        repository.delete_autobio_summary(conn, start_date, end_date)
    return redirect(url_for("autobio_index"))


@app.get("/autobio/summary/<start_date>/<end_date>/export/<fmt>")
def autobio_summary_export(start_date, end_date, fmt):
    conn = get_conn()
    summary = repository.get_autobio_summary(conn, start_date, end_date)
    if summary is None:
        abort(404)

    content = export.content_for_autobio_summary(summary)
    try:
        mimetype, data = export.export(content, fmt)
    except ValueError:
        abort(404)

    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"autobio-{start_date}-to-{end_date}.{fmt}",
    )


# ---------------------------------------------------------------------
# App settings — labeling/narrative/UI language (§4.2 and after), plus
# the "Connect your accounts" section below: self-hosting used to mean
# hand-editing files under secrets/ per README.md's setup walkthrough;
# these routes let that happen from the browser instead. They write to
# the exact same files/paths google_auth.py and llm.py already read from,
# so nothing about how credentials are *used* changes — only how they get
# there in the first place.
# ---------------------------------------------------------------------


def _google_configured() -> bool:
    return CLIENT_SECRET_PATH.exists()


def _effective_originals_dir(conn) -> Path:
    """Where new imports save their originals — the user's configured
    `originals_dir` app_setting if they've set one, else
    DEFAULT_ORIGINALS_DIR. Both Google-Photos and local-device imports
    call this (see import_continue()/import_local_upload()'s run()
    functions) — they've always shared one folder, this just makes which
    folder configurable instead of hardcoded."""
    configured = repository.get_app_settings(conn)["originals_dir"]
    return Path(configured) if configured else DEFAULT_ORIGINALS_DIR


def _llm_configured() -> bool:
    """Whether *some* provider is actually usable right now — checks the
    same env-var-then-file resolution llm.py itself uses (not just "does a
    key file exist"), so an env-var-only setup (e.g. Docker) still reports
    correctly configured.

    Explicitly passes this module's own ANTHROPIC_KEY_PATH/OPENAI_KEY_PATH
    rather than letting llm.py compute its own defaults — those two
    happen to land on the same real path today, but only the module-level
    constants here are what the save routes above actually write to (and
    what tests patch); calling resolve_provider_and_key() with no
    arguments would silently re-derive its own path and ignore both."""
    try:
        llm.resolve_provider_and_key(
            anthropic_key_path=ANTHROPIC_KEY_PATH, openai_key_path=OPENAI_KEY_PATH
        )
        return True
    except llm.LLMNotConfigured:
        return False


@app.route("/settings")
def settings_page():
    conn = get_conn()
    return render_template(
        "settings.html",
        settings=repository.get_app_settings(conn),
        languages=repository.SPEECH_LANGUAGES,
        narrative_languages=repository.LANGUAGES,
        ui_languages=repository.LANGUAGES,
        google_configured=_google_configured(),
        anthropic_configured=ANTHROPIC_KEY_PATH.exists(),
        openai_configured=OPENAI_KEY_PATH.exists(),
        default_originals_dir=str(DEFAULT_ORIGINALS_DIR),
    )


@app.route("/settings/connect")
def settings_connect_page():
    conn = get_conn()
    return render_template(
        "settings_connect.html",
        settings=repository.get_app_settings(conn),
        google_configured=_google_configured(),
        google_email=google_auth.get_cached_email(TOKEN_CACHE_PATH),
        anthropic_configured=ANTHROPIC_KEY_PATH.exists(),
        openai_configured=OPENAI_KEY_PATH.exists(),
    )


@app.post("/settings/speech-language")
def settings_speech_language_save():
    conn = get_conn()
    language = (request.form.get("speech_language") or "").strip()
    if not language:
        return render_template("error.html", message="Pick a language."), 400
    with conn:
        repository.set_speech_language(conn, language)
    return redirect(url_for("settings_page"))


@app.post("/settings/narrative-language")
def settings_narrative_language_save():
    conn = get_conn()
    language = (request.form.get("narrative_language") or "").strip()
    if not language:
        return render_template("error.html", message="Pick a language."), 400
    with conn:
        repository.set_narrative_language(conn, language)
    return redirect(url_for("settings_page"))


@app.post("/settings/ui-language")
def settings_ui_language_save():
    conn = get_conn()
    language = (request.form.get("ui_language") or "").strip()
    if not language:
        return render_template("error.html", message="Pick a language."), 400
    with conn:
        repository.set_ui_language(conn, language)
    return redirect(url_for("settings_page"))


def _write_secret_file(path: Path, content: str) -> None:
    """Writes a credential file the way a self-hoster used to have to by
    hand — same location, same format — then locks it down to
    owner-read/write only. `secrets/` may not exist yet on a truly fresh
    checkout (it's gitignored, not part of the repo), so this creates it
    rather than assuming README.md's manual setup already ran."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    os.chmod(path, 0o600)


@app.post("/settings/google-credentials")
def settings_google_credentials_save():
    upload = request.files.get("client_secret_file")
    if not upload or not upload.filename:
        return render_template("error.html", message="Choose the JSON file Google Cloud Console gave you."), 400
    try:
        raw = upload.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        return render_template("error.html", message="That file isn't valid text — is it really the downloaded JSON file?"), 400
    if not raw:
        return render_template("error.html", message="That file is empty."), 400
    try:
        data = json.loads(raw)
        block = data.get("installed") or data.get("web")
        if not block or "client_id" not in block or "client_secret" not in block:
            raise ValueError("missing client_id/client_secret")
    except (json.JSONDecodeError, ValueError):
        return render_template(
            "error.html",
            message=(
                "That didn't look like a Google OAuth client JSON file — it should "
                'have an "installed" (or "web") section with client_id and '
                "client_secret. Re-download it from Google Cloud Console and upload "
                "that file."
            ),
        ), 400
    _write_secret_file(CLIENT_SECRET_PATH, raw)
    return redirect(url_for("settings_connect_page"))


@app.post("/settings/anthropic-key")
def settings_anthropic_key_save():
    key = (request.form.get("anthropic_api_key") or "").strip()
    if not key:
        return render_template("error.html", message="Paste an Anthropic API key."), 400
    _write_secret_file(ANTHROPIC_KEY_PATH, key + "\n")
    return redirect(url_for("settings_connect_page"))


@app.post("/settings/openai-key")
def settings_openai_key_save():
    key = (request.form.get("openai_api_key") or "").strip()
    if not key:
        return render_template("error.html", message="Paste an OpenAI API key."), 400
    _write_secret_file(OPENAI_KEY_PATH, key + "\n")
    return redirect(url_for("settings_connect_page"))


@app.post("/settings/llm-provider")
def settings_llm_provider_save():
    conn = get_conn()
    provider = (request.form.get("llm_provider") or "").strip()
    if provider not in ("", "anthropic", "openai"):
        abort(400)
    with conn:
        repository.set_llm_provider(conn, provider)
    return redirect(url_for("settings_connect_page"))


@app.post("/settings/originals-dir")
def settings_originals_dir_save():
    """Where future imports save their originals (§5's LocalStorageAdapter
    root_dir) — see _effective_originals_dir(), which both import routes
    call instead of the old hardcoded DEFAULT_ORIGINALS_DIR.

    Only affects imports from here on: existing photos' original_storage_path
    rows already point at wherever they were saved and are not moved — a
    changed setting doesn't retroactively relocate anything already on
    disk. The settings page copy says as much, so this isn't a surprise.
    """
    conn = get_conn()
    raw = (request.form.get("originals_dir") or "").strip()
    if not raw:
        # Empty clears it back to the default — same convention as the
        # other app_settings fields (llm_provider's "" == auto-detect).
        with conn:
            repository.set_originals_dir(conn, "")
        return redirect(url_for("settings_page"))

    path = Path(raw).expanduser()
    if not path.is_absolute():
        return render_template("error.html", message="Enter an absolute path."), 400
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return render_template(
            "error.html", message=f"Can't use that folder: {exc}"
        ), 400
    with conn:
        repository.set_originals_dir(conn, str(path))
    return redirect(url_for("settings_page"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local labeling UI.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Defaults to localhost only. Pass 0.0.0.0 to reach it from other devices "
        "on your network — that exposes your photo library to that network.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.config["DB_PATH"] = args.db
    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/ingest.py first.")
        raise SystemExit(1)

    # Apply any pending migrations on startup, matching scripts/ingest.py.
    # Without this, a database created before a schema change fails at
    # request time with a bare "no such column" 500 rather than being
    # brought up to date.
    conn = db.connect(args.db)
    applied = db.migrate(conn)
    conn.close()
    if applied:
        print(f"Applied migration(s): {', '.join(applied)}")

    print(f"Labeling UI on http://{args.host}:{args.port}  (database: {args.db})")
    # threaded=True: an import (§ 'Import from Google Photos') blocks its
    # request for as long as OAuth consent + Picker selection + ingestion
    # take — Flask's dev server is single-threaded by default, which would
    # freeze every other open tab for that whole time otherwise.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

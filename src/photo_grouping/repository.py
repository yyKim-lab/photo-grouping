"""Database access for the ingestion pipeline — all SQL lives here.

Keeps ingestion.py readable as a description of the §3 flow rather than a
pile of INSERT statements, and gives the tests one place to exercise
persistence independently of the Picker/face/GPS machinery.

Embeddings are stored as the schema documents: a packed little-endian
float32 array (stdlib `array`, no numpy dependency at the storage layer).
face_recognition produces float64; the narrowing is harmless here, since
float32 carries ~7 decimal digits and face distances are compared against
thresholds around 0.3.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from datetime import datetime
from typing import Optional, Sequence


def encode_embedding(values: Sequence[float]) -> bytes:
    return array("f", (float(v) for v in values)).tobytes()


def decode_embedding(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else value


# ---------------------------------------------------------------------
# Photo
# ---------------------------------------------------------------------


def photo_id_for_picker_media_id(conn: sqlite3.Connection, picker_media_id: str) -> Optional[int]:
    """Ingestion is re-runnable: picker_media_id is UNIQUE, so an already
    imported photo is skipped rather than duplicated or failing the run."""
    row = conn.execute(
        "SELECT id FROM photo WHERE picker_media_id = ?", (picker_media_id,)
    ).fetchone()
    return row["id"] if row else None


def insert_photo(
    conn: sqlite3.Connection,
    *,
    picker_media_id: str,
    taken_at: datetime | str,
    original_filename: str,
    original_storage_backend: str,
    original_storage_path: str,
    gps_lat: Optional[float] = None,
    gps_lng: Optional[float] = None,
) -> int:
    """`original_filename` is the name Google reports, which is not
    necessarily the stored file's basename — storage adapters dedup-suffix
    colliding names. Takeout sidecars key on the original, so it's stored
    separately (see migration 0002)."""
    cursor = conn.execute(
        """
        INSERT INTO photo (
            picker_media_id, taken_at, gps_lat, gps_lng,
            original_filename, original_storage_backend, original_storage_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            picker_media_id,
            _iso(taken_at),
            gps_lat,
            gps_lng,
            original_filename,
            original_storage_backend,
            original_storage_path,
        ),
    )
    return cursor.lastrowid


def set_photo_gps(conn: sqlite3.Connection, photo_id: int, lat: float, lng: float) -> None:
    """Used by the Takeout backfill, which runs separately from ingestion
    (the Picker API can't supply GPS — see README's platform-constraint
    note). Never touches gps_override_* : those are the user's manual
    corrections (§4.4) and must survive a re-run of the backfill."""
    conn.execute(
        "UPDATE photo SET gps_lat = ?, gps_lng = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (lat, lng, photo_id),
    )


# ---------------------------------------------------------------------
# Face clusters / instances
# ---------------------------------------------------------------------


def load_face_cluster_centroids(conn: sqlite3.Connection) -> list[tuple[int, list[float], int]]:
    """Returns (cluster_id, centroid, member_count) for every face cluster,
    rebuilding centroids from member embeddings — see face_clustering.py's
    docstring for why these aren't stored."""
    rows = conn.execute(
        "SELECT face_cluster_id, embedding FROM face_instance "
        "WHERE false_positive_at IS NULL ORDER BY face_cluster_id"
    ).fetchall()

    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        cluster_id = row["face_cluster_id"]
        embedding = decode_embedding(row["embedding"])
        if cluster_id not in sums:
            sums[cluster_id] = list(embedding)
            counts[cluster_id] = 1
        else:
            sums[cluster_id] = [a + b for a, b in zip(sums[cluster_id], embedding)]
            counts[cluster_id] += 1

    return [
        (cluster_id, [total / counts[cluster_id] for total in sums[cluster_id]], counts[cluster_id])
        for cluster_id in sums
    ]


def insert_face_cluster(
    conn: sqlite3.Connection,
    *,
    representative_photo_id: Optional[int] = None,
    suggested_name: Optional[str] = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO face_cluster (status, representative_photo_id, suggested_name) "
        "VALUES ('unlabeled', ?, ?)",
        (representative_photo_id, suggested_name),
    )
    return cursor.lastrowid


def insert_face_instance(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    face_cluster_id: int,
    bounding_box: dict,
    embedding: Sequence[float],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO face_instance (photo_id, face_cluster_id, bounding_box, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (photo_id, face_cluster_id, json.dumps(bounding_box), encode_embedding(embedding)),
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------
# Seed faces (§4.5) — matched against during ingestion (§3 step 5)
# ---------------------------------------------------------------------


def load_seed_faces(conn: sqlite3.Connection) -> list[tuple[int, str, list[float]]]:
    return [
        (row["id"], row["name"], decode_embedding(row["embedding"]))
        for row in conn.execute("SELECT id, name, embedding FROM seed_face")
    ]


def insert_seed_face(
    conn: sqlite3.Connection, *, name: str, embedding: Sequence[float], source: str = "screenshot_import"
) -> int:
    cursor = conn.execute(
        "INSERT INTO seed_face (name, embedding, source) VALUES (?, ?, ?)",
        (name.strip(), encode_embedding(embedding), source),
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------
# Location clusters
# ---------------------------------------------------------------------


def load_location_clusters(conn: sqlite3.Connection) -> list[tuple[int, float, float, int]]:
    """Returns (id, centroid_lat, centroid_lng, member_count)."""
    rows = conn.execute(
        """
        SELECT lc.id, lc.centroid_lat, lc.centroid_lng, COUNT(pl.photo_id) AS member_count
        FROM location_cluster lc
        LEFT JOIN photo_location pl ON pl.location_cluster_id = lc.id
        GROUP BY lc.id
        """
    ).fetchall()
    # Coordinate-less clusters (see NO_COORDINATES below) are excluded —
    # they're never spatially clustered, so they must never enter the
    # nearest-centroid candidate list assign_or_create() compares against.
    # A real GPS point vs. the sentinel would compute some enormous
    # haversine distance and likely never get chosen anyway, but excluding
    # them explicitly is correct by construction, not by luck.
    return [
        (r["id"], r["centroid_lat"], r["centroid_lng"], max(r["member_count"], 1))
        for r in rows
        if r["centroid_lat"] != NO_COORDINATES
    ]


# A place with no real coordinates (§4.4-adjacent: a location the user
# knows by name but that isn't a mappable point — "우리집", a private
# venue, anywhere the geocoder just doesn't know) — centroid_lat/lng are
# NOT NULL in the schema (migration 0001) and every existing consumer
# (map links, the nearest-centroid clustering algorithm) assumes real
# coordinates, so this uses a sentinel pair rather than a nullable-column
# migration: SQLite can only relax a NOT NULL/CHECK constraint by
# rebuilding the table and re-pointing every foreign key into it, which
# migration 0004's own docstring already flagged as too risky to do for
# a flag — doing that against a user's real, irreplaceable photo database
# is not a trade worth making for this. 999.0 is outside the valid
# latitude (-90..90) and longitude (-180..180) ranges, so it can never
# collide with a real GPS reading.
NO_COORDINATES = 999.0


def insert_location_cluster(conn: sqlite3.Connection, *, lat: float, lng: float) -> int:
    cursor = conn.execute(
        "INSERT INTO location_cluster (status, centroid_lat, centroid_lng) VALUES ('unlabeled', ?, ?)",
        (lat, lng),
    )
    return cursor.lastrowid


def update_location_cluster_centroid(
    conn: sqlite3.Connection, cluster_id: int, lat: float, lng: float
) -> None:
    conn.execute(
        """
        UPDATE location_cluster
        SET centroid_lat = ?, centroid_lng = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (lat, lng, cluster_id),
    )


def set_photo_location(conn: sqlite3.Connection, *, photo_id: int, location_cluster_id: int) -> None:
    """Won't clobber a manual correction: a photo_location row with
    is_manual_override set is the user's own assignment (§4.4), and
    re-clustering must respect it."""
    existing = conn.execute(
        "SELECT is_manual_override FROM photo_location WHERE photo_id = ?", (photo_id,)
    ).fetchone()
    if existing and existing["is_manual_override"]:
        return
    conn.execute(
        """
        INSERT INTO photo_location (photo_id, location_cluster_id) VALUES (?, ?)
        ON CONFLICT(photo_id) DO UPDATE SET
            location_cluster_id = excluded.location_cluster_id,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (photo_id, location_cluster_id),
    )


# ---------------------------------------------------------------------
# Reads for downstream features
# ---------------------------------------------------------------------


def load_photos_for_anomaly_check(conn: sqlite3.Connection) -> list[dict]:
    """Photos with an effective location and timestamp, plus the face
    clusters appearing in each — the input gps_anomalies.py needs.

    Uses the *_override columns in preference to the raw EXIF/Takeout
    values throughout, since a user correction (§4.4) is by definition
    more trustworthy than the data that needed correcting.
    """
    rows = conn.execute(
        """
        SELECT
            p.id,
            COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at,
            COALESCE(p.gps_override_lat, p.gps_lat)   AS effective_lat,
            COALESCE(p.gps_override_lng, p.gps_lng)   AS effective_lng
        FROM photo p
        WHERE effective_lat IS NOT NULL AND effective_lng IS NOT NULL
        ORDER BY effective_taken_at
        """
    ).fetchall()

    face_map: dict[int, set[int]] = {}
    for row in conn.execute(
        "SELECT photo_id, face_cluster_id FROM face_instance WHERE false_positive_at IS NULL"
    ):
        face_map.setdefault(row["photo_id"], set()).add(row["face_cluster_id"])

    return [
        {
            "photo_id": row["id"],
            "taken_at": row["effective_taken_at"],
            "lat": row["effective_lat"],
            "lng": row["effective_lng"],
            "face_cluster_ids": frozenset(face_map.get(row["id"], set())),
        }
        for row in rows
    ]


def photos_missing_gps(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """(photo_id, original_filename) for photos with no location yet — what
    the Takeout backfill looks up, since sidecars key on the original
    filename rather than on wherever the file ended up on disk."""
    rows = conn.execute(
        """
        SELECT id, original_filename
        FROM photo
        WHERE gps_lat IS NULL AND gps_override_lat IS NULL AND original_filename IS NOT NULL
        """
    ).fetchall()
    return [(row["id"], row["original_filename"]) for row in rows]


# ---------------------------------------------------------------------
# Labeling UI (§4.1 browsing, §4.2 queue, §4.3 cluster detail)
# ---------------------------------------------------------------------

# §4.2: a not_sure cluster returns to the queue once it has grown by this
# many instances since the user last saw it. Decided value, up from the
# spec's earlier +3 placeholder.
REGROW_THRESHOLD = 5


def load_labeling_queue(conn: sqlite3.Connection) -> list[dict]:
    """§4.2's queue: every unlabeled cluster, plus not_sure clusters that
    have grown by >= REGROW_THRESHOLD instances since last shown.

    Faces and places are returned in one list with a `kind` discriminator,
    because the queue itself is one stream — the card just swaps its prompt
    between "Do you know this person?" and "Do you know this place?".

    Ordering puts the biggest clusters first: naming the person who appears
    in 27 photos resolves far more of the library per decision than naming
    a one-off, so the queue front-loads the highest-value questions.
    """
    face_rows = conn.execute(
        f"""
        SELECT fc.id, fc.status, fc.last_shown_at, fc.suggested_name,
               COUNT(fi.id) AS instance_count,
               fc.instance_count_at_last_shown
        FROM face_cluster fc
        JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
        WHERE fc.status IN ('unlabeled', 'not_sure') AND fc.excluded_at IS NULL
        GROUP BY fc.id
        HAVING fc.status = 'unlabeled'
            OR instance_count - fc.instance_count_at_last_shown >= {REGROW_THRESHOLD}
        """
    ).fetchall()

    place_rows = conn.execute(
        f"""
        SELECT lc.id, lc.status, lc.last_shown_at, lc.centroid_lat, lc.centroid_lng,
               COUNT(pl.photo_id) AS instance_count,
               lc.instance_count_at_last_shown
        FROM location_cluster lc
        JOIN photo_location pl ON pl.location_cluster_id = lc.id
        WHERE lc.status IN ('unlabeled', 'not_sure') AND lc.excluded_at IS NULL
        GROUP BY lc.id
        HAVING lc.status = 'unlabeled'
            OR instance_count - lc.instance_count_at_last_shown >= {REGROW_THRESHOLD}
        """
    ).fetchall()

    items = [
        {
            "kind": "face",
            "id": r["id"],
            "status": r["status"],
            "instance_count": r["instance_count"],
            "suggested_name": r["suggested_name"],
        }
        for r in face_rows
    ] + [
        {
            "kind": "place",
            "id": r["id"],
            "status": r["status"],
            "instance_count": r["instance_count"],
            "centroid_lat": r["centroid_lat"],
            "centroid_lng": r["centroid_lng"],
        }
        for r in place_rows
    ]
    return sorted(items, key=lambda item: -item["instance_count"])


def photos_with_nothing_detected(conn: sqlite3.Connection) -> list[dict]:
    """Photos with no detected face *and* no location at all — a real gap
    found in practice: load_labeling_queue() above is entirely cluster-
    based (a face_cluster or location_cluster row), so a photo that never
    produced either (no face detected, and no GPS to cluster by) never
    appears in the queue, in People, or in Places — it's structurally
    invisible with no prompt to do anything about it. There's no cluster
    to "label" here, so this isn't added as a queue item; instead
    queue_empty.html links each one straight to its own photo_detail
    page, which already has a manual "type a place name" flow that works
    without GPS (see set_photo_location_by_name)."""
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT p.id AS photo_id,
                   COALESCE(p.taken_at_override, p.taken_at) AS taken_at
            FROM photo p
            LEFT JOIN photo_location pl ON pl.photo_id = p.id
            LEFT JOIN face_instance fi ON fi.photo_id = p.id AND fi.false_positive_at IS NULL
            WHERE pl.photo_id IS NULL AND fi.id IS NULL
            GROUP BY p.id
            ORDER BY taken_at DESC
            """
        )
    ]


def representative_face(conn: sqlite3.Connection, face_cluster_id: int) -> Optional[dict]:
    """The face instance to show on a cluster's card: the physically
    largest one, since a bigger crop is a clearer look at the person than
    an arbitrary or merely-first pick."""
    row = conn.execute(
        """
        SELECT fi.id, fi.photo_id, fi.bounding_box, p.original_storage_path
        FROM face_instance fi
        JOIN photo p ON p.id = fi.photo_id
        WHERE fi.face_cluster_id = ? AND fi.false_positive_at IS NULL
        """,
        (face_cluster_id,),
    ).fetchall()
    if not row:
        return None
    best = max(row, key=lambda r: json.loads(r["bounding_box"])["width"])
    return {
        "face_instance_id": best["id"],
        "photo_id": best["photo_id"],
        "bounding_box": json.loads(best["bounding_box"]),
        "path": best["original_storage_path"],
    }


def face_instances_in_cluster(conn: sqlite3.Connection, face_cluster_id: int) -> list[dict]:
    return [
        {
            "face_instance_id": r["id"],
            "photo_id": r["photo_id"],
            "bounding_box": json.loads(r["bounding_box"]),
            "path": r["original_storage_path"],
            "filename": r["original_filename"],
            "taken_at": r["effective_taken_at"],
        }
        for r in conn.execute(
            """
            SELECT fi.id, fi.photo_id, fi.bounding_box,
                   p.original_storage_path, p.original_filename,
                   COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at
            FROM face_instance fi
            JOIN photo p ON p.id = fi.photo_id
            WHERE fi.face_cluster_id = ? AND fi.false_positive_at IS NULL
            ORDER BY effective_taken_at
            """,
            (face_cluster_id,),
        )
    ]


def photos_in_face_cluster(conn: sqlite3.Connection, face_cluster_id: int) -> list[dict]:
    """The distinct photos a person appears in — for browsing an album as
    pictures rather than as cropped faces (§4.1's actual intent; cropped
    faces are for split/merge judgement calls, see face_instances_in_cluster).

    DISTINCT because a person can have more than one face_instance in the
    same photo in principle (a mirror, a photo of a photo) — rare, but a
    photo should appear once in the album regardless.
    """
    return [
        {
            "photo_id": r["id"],
            "path": r["original_storage_path"],
            "filename": r["original_filename"],
            "taken_at": r["effective_taken_at"],
        }
        for r in conn.execute(
            """
            SELECT DISTINCT p.id, p.original_storage_path, p.original_filename,
                   COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at
            FROM face_instance fi
            JOIN photo p ON p.id = fi.photo_id
            WHERE fi.face_cluster_id = ? AND fi.false_positive_at IS NULL
            ORDER BY effective_taken_at
            """,
            (face_cluster_id,),
        )
    ]


def photos_in_location_cluster(conn: sqlite3.Connection, location_cluster_id: int) -> list[dict]:
    return [
        {
            "photo_id": r["id"],
            "path": r["original_storage_path"],
            "filename": r["original_filename"],
            "taken_at": r["effective_taken_at"],
            "lat": r["effective_lat"],
            "lng": r["effective_lng"],
        }
        for r in conn.execute(
            """
            SELECT p.id, p.original_storage_path, p.original_filename,
                   COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at,
                   COALESCE(p.gps_override_lat, p.gps_lat) AS effective_lat,
                   COALESCE(p.gps_override_lng, p.gps_lng) AS effective_lng
            FROM photo_location pl
            JOIN photo p ON p.id = pl.photo_id
            WHERE pl.location_cluster_id = ?
            ORDER BY effective_taken_at
            """,
            (location_cluster_id,),
        )
    ]


def _table_for(kind: str) -> str:
    if kind not in ("face", "place"):
        raise ValueError(f"Unknown cluster kind: {kind!r}")
    return "face_cluster" if kind == "face" else "location_cluster"


def cluster_row(conn: sqlite3.Connection, kind: str, cluster_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM {_table_for(kind)} WHERE id = ?", (cluster_id,)
    ).fetchone()


def name_cluster(conn: sqlite3.Connection, kind: str, cluster_id: int, name: str) -> None:
    """§4.2 'Yes' -> the cluster becomes a named album."""
    conn.execute(
        f"""
        UPDATE {_table_for(kind)}
        SET status = 'named', name = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (name.strip(), cluster_id),
    )


def mark_cluster_not_sure(conn: sqlite3.Connection, kind: str, cluster_id: int) -> None:
    """§4.2 'Not sure' -> stays in Uncategorized, and won't return to the
    queue until it has grown by REGROW_THRESHOLD instances. Recording the
    count *now* is what makes that comparison meaningful later."""
    table = _table_for(kind)
    if kind == "face":
        count_sql = (
            "SELECT COUNT(*) AS c FROM face_instance "
            "WHERE face_cluster_id = ? AND false_positive_at IS NULL"
        )
    else:
        count_sql = "SELECT COUNT(*) AS c FROM photo_location WHERE location_cluster_id = ?"
    current = conn.execute(count_sql, (cluster_id,)).fetchone()["c"]

    conn.execute(
        f"""
        UPDATE {table}
        SET status = 'not_sure',
            last_shown_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            instance_count_at_last_shown = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (current, cluster_id),
    )


def load_albums(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """§4.1: named clusters, grouped by name. Two clusters that were named
    the same thing (e.g. after a split the user never merged back) present
    as one album, which is what the spec's 'Albums = clusters where status
    = named, grouped by name' asks for."""
    faces = conn.execute(
        """
        SELECT fc.name, fc.id, COUNT(fi.id) AS instance_count
        FROM face_cluster fc
        LEFT JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
        WHERE fc.status = 'named' AND fc.excluded_at IS NULL
        GROUP BY fc.id
        ORDER BY fc.name
        """
    ).fetchall()
    places = conn.execute(
        """
        SELECT lc.name, lc.id, COUNT(pl.photo_id) AS instance_count
        FROM location_cluster lc
        LEFT JOIN photo_location pl ON pl.location_cluster_id = lc.id
        WHERE lc.status = 'named' AND lc.excluded_at IS NULL
        GROUP BY lc.id
        ORDER BY lc.name
        """
    ).fetchall()

    def group(rows):
        grouped: dict[str, dict] = {}
        for row in rows:
            entry = grouped.setdefault(row["name"], {"name": row["name"], "ids": [], "instance_count": 0})
            entry["ids"].append(row["id"])
            entry["instance_count"] += row["instance_count"]
        return sorted(grouped.values(), key=lambda e: -e["instance_count"])

    return {"people": group(faces), "places": group(places)}


def load_uncategorized(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """§4.1: unlabeled + not_sure clusters, with the status kept so the UI
    can badge not_sure ones distinctly."""
    faces = conn.execute(
        """
        SELECT fc.id, fc.status, COUNT(fi.id) AS instance_count
        FROM face_cluster fc
        JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
        WHERE fc.status IN ('unlabeled', 'not_sure') AND fc.excluded_at IS NULL
        GROUP BY fc.id
        ORDER BY instance_count DESC
        """
    ).fetchall()
    places = conn.execute(
        """
        SELECT lc.id, lc.status, lc.centroid_lat, lc.centroid_lng, COUNT(pl.photo_id) AS instance_count
        FROM location_cluster lc
        JOIN photo_location pl ON pl.location_cluster_id = lc.id
        WHERE lc.status IN ('unlabeled', 'not_sure') AND lc.excluded_at IS NULL
        GROUP BY lc.id
        ORDER BY instance_count DESC
        """
    ).fetchall()
    return {
        "people": [dict(r) for r in faces],
        "places": [dict(r) for r in places],
    }


# ---------------------------------------------------------------------
# Per-photo metadata edits (§4.4)
# ---------------------------------------------------------------------


def photo_detail(conn: sqlite3.Connection, photo_id: int) -> Optional[dict]:
    """Everything the §4.4 single-photo edit view needs: the photo itself
    (with its overrides), every face detected in it with its current
    cluster assignment, and its current location assignment — one call
    rather than the view assembling several."""
    photo = conn.execute("SELECT * FROM photo WHERE id = ?", (photo_id,)).fetchone()
    if photo is None:
        return None

    faces = [
        {
            "face_instance_id": r["id"],
            "bounding_box": json.loads(r["bounding_box"]),
            "cluster_id": r["face_cluster_id"],
            "cluster_name": r["name"],
            "cluster_status": r["status"],
            "is_manual_override": bool(r["is_manual_override"]),
        }
        for r in conn.execute(
            """
            SELECT fi.id, fi.bounding_box, fi.face_cluster_id, fi.is_manual_override,
                   fc.name, fc.status
            FROM face_instance fi
            JOIN face_cluster fc ON fc.id = fi.face_cluster_id
            WHERE fi.photo_id = ? AND fi.false_positive_at IS NULL
            ORDER BY fi.id
            """,
            (photo_id,),
        )
    ]

    location_row = conn.execute(
        """
        SELECT lc.id, lc.name, lc.centroid_lat, lc.centroid_lng, pl.is_manual_override
        FROM photo_location pl JOIN location_cluster lc ON lc.id = pl.location_cluster_id
        WHERE pl.photo_id = ?
        """,
        (photo_id,),
    ).fetchone()

    return {
        "photo": dict(photo),
        "faces": faces,
        "location": dict(location_row) if location_row else None,
        "events": events_for_photo(conn, photo_id),
    }


def reassign_face(conn: sqlite3.Connection, face_instance_id: int, name: str) -> int:
    """§4.4: 'allow reassign to a different cluster'. `name` matches an
    existing named cluster if one exists (case-sensitive, matching how
    names are compared everywhere else in the app), else creates a new
    named cluster for it — typing a name is how the rest of the app already
    creates people (§4.2's queue), so reassignment reuses the same idea
    rather than requiring the user to pick from a list of face crops.

    Marks the moved instance is_manual_override so re-clustering leaves it
    alone (§4.4's own requirement).
    """
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")

    existing = conn.execute(
        "SELECT id FROM face_cluster WHERE name = ? AND status = 'named'", (name,)
    ).fetchone()
    target_cluster_id = existing["id"] if existing else insert_face_cluster(conn)
    if existing is None:
        name_cluster(conn, "face", target_cluster_id, name)

    conn.execute(
        """
        UPDATE face_instance
        SET face_cluster_id = ?, is_manual_override = 1,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (target_cluster_id, face_instance_id),
    )
    return target_cluster_id


def mark_face_false_positive(conn: sqlite3.Connection, face_instance_id: int) -> None:
    """§4.4: 'remove (mark as false-positive detection)'. A timestamp flag
    rather than a delete — see migration 0006 for why — so every query that
    aggregates or lists faces must filter false_positive_at IS NULL."""
    conn.execute(
        "UPDATE face_instance SET false_positive_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (face_instance_id,),
    )


def restore_false_positive(conn: sqlite3.Connection, face_instance_id: int) -> None:
    conn.execute(
        "UPDATE face_instance SET false_positive_at = NULL WHERE id = ?", (face_instance_id,)
    )


def add_manual_face(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    bounding_box: dict,
    embedding: Sequence[float],
    name: str,
) -> int:
    """§4.4: 'manually add a face/cluster if detection missed someone'.
    Always assigns straight to a named cluster (existing or new) rather
    than leaving it unlabeled — the user is looking at the photo and
    naming the person in the same motion, so there's no reason to route
    it through the labeling queue afterward."""
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")

    existing = conn.execute(
        "SELECT id FROM face_cluster WHERE name = ? AND status = 'named'", (name,)
    ).fetchone()
    cluster_id = existing["id"] if existing else insert_face_cluster(conn, representative_photo_id=photo_id)
    if existing is None:
        name_cluster(conn, "face", cluster_id, name)

    face_instance_id = insert_face_instance(
        conn, photo_id=photo_id, face_cluster_id=cluster_id, bounding_box=bounding_box, embedding=embedding
    )
    conn.execute(
        "UPDATE face_instance SET is_manual_override = 1 WHERE id = ?", (face_instance_id,)
    )
    return cluster_id


def override_taken_at(conn: sqlite3.Connection, photo_id: int, taken_at: str) -> None:
    """§4.4: date/time override. `taken_at` is an ISO-8601 string; the
    caller (the web form) is responsible for parsing user input into one."""
    conn.execute(
        """
        UPDATE photo
        SET taken_at_override = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (taken_at, photo_id),
    )


def override_photo_location(conn: sqlite3.Connection, photo_id: int, lat: float, lng: float) -> int:
    """§4.4: location override. Sets gps_override_lat/lng (which
    §3/§4.4 says supersedes raw EXIF everywhere downstream), then
    immediately (re)assigns the photo to a location cluster using the same
    nearest-centroid logic ingestion uses — the override should take visible
    effect right away, not wait for the next batch clustering pass.

    Unlike set_photo_location(), this *does* overwrite an existing manual
    assignment: this call is itself the manual correction, so the usual
    "don't clobber a manual override" guard would block the very edit being
    requested.
    """
    from . import location_clustering

    conn.execute(
        """
        UPDATE photo
        SET gps_override_lat = ?, gps_override_lng = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (lat, lng, photo_id),
    )

    clusters = [
        location_clustering.LocationClusterCandidate(centroid_lat=clat, centroid_lng=clng, count=count)
        for _cid, clat, clng, count in load_location_clusters(conn)
    ]
    cluster_ids = [cid for cid, *_ in load_location_clusters(conn)]

    before = len(clusters)
    chosen = location_clustering.assign_or_create(lat, lng, clusters, photo_index=photo_id)
    index = clusters.index(chosen)

    if len(clusters) > before:
        cluster_id = insert_location_cluster(conn, lat=chosen.centroid_lat, lng=chosen.centroid_lng)
    else:
        cluster_id = cluster_ids[index]
        update_location_cluster_centroid(conn, cluster_id, chosen.centroid_lat, chosen.centroid_lng)

    conn.execute(
        """
        INSERT INTO photo_location (photo_id, location_cluster_id, is_manual_override)
        VALUES (?, ?, 1)
        ON CONFLICT(photo_id) DO UPDATE SET
            location_cluster_id = excluded.location_cluster_id,
            is_manual_override = 1,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (photo_id, cluster_id),
    )
    return cluster_id


def set_photo_location_by_name_without_coordinates(
    conn: sqlite3.Connection, photo_id: int, name: str
) -> int:
    """For a place the user knows by name but that isn't a mappable point
    (geocoding found nothing) — labels the photo directly with `name`
    instead of erroring, using the NO_COORDINATES sentinel (see that
    constant's docstring). Reuses an existing coordinate-less cluster with
    the exact same name rather than creating a duplicate every time, the
    same match-by-name idea as get_or_create_event(); real-GPS clusters
    are never candidates here since there's no coordinate to match by."""
    name = name.strip()
    if not name:
        raise ValueError("Place name cannot be empty.")

    existing = conn.execute(
        "SELECT id FROM location_cluster WHERE name = ? AND centroid_lat = ?",
        (name, NO_COORDINATES),
    ).fetchone()
    if existing:
        cluster_id = existing["id"]
    else:
        cursor = conn.execute(
            "INSERT INTO location_cluster (status, name, centroid_lat, centroid_lng) "
            "VALUES ('named', ?, ?, ?)",
            (name, NO_COORDINATES, NO_COORDINATES),
        )
        cluster_id = cursor.lastrowid

    conn.execute(
        """
        INSERT INTO photo_location (photo_id, location_cluster_id, is_manual_override)
        VALUES (?, ?, 1)
        ON CONFLICT(photo_id) DO UPDATE SET
            location_cluster_id = excluded.location_cluster_id,
            is_manual_override = 1,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (photo_id, cluster_id),
    )
    return cluster_id


# ---------------------------------------------------------------------
# Split / merge (§4.3)
# ---------------------------------------------------------------------


def _log_cluster_event(
    conn: sqlite3.Connection,
    *,
    cluster_type: str,
    event_type: str,
    source_ids: list,
    resulting_ids: list,
) -> None:
    conn.execute(
        """
        INSERT INTO cluster_event (cluster_type, event_type, source_cluster_ids, resulting_cluster_ids)
        VALUES (?, ?, ?, ?)
        """,
        (cluster_type, event_type, json.dumps(source_ids), json.dumps(resulting_ids)),
    )


def similar_face_clusters(
    conn: sqlite3.Connection, cluster_id: int, limit: int = 6
) -> list[dict]:
    """Other face clusters ranked by centroid similarity — merge candidates.

    Surfacing ranked candidates on a cluster's own page is what makes merge
    usable in practice: the clustering threshold is deliberately
    conservative (see README "Face embedding"), so a person being split in
    two is expected, and the other half is almost always the top candidate
    here.
    """
    from . import face_clustering

    centroids = {cid: centroid for cid, centroid, _count in load_face_cluster_centroids(conn)}
    if cluster_id not in centroids:
        return []
    target = centroids[cluster_id]

    rows = {
        r["id"]: r
        for r in conn.execute(
            """
            SELECT fc.id, fc.name, fc.status, COUNT(fi.id) AS instance_count
            FROM face_cluster fc
            JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
            WHERE fc.excluded_at IS NULL
            GROUP BY fc.id
            """
        )
    }

    scored = []
    for other_id, centroid in centroids.items():
        if other_id == cluster_id or other_id not in rows:
            continue
        scored.append(
            {
                "id": other_id,
                "name": rows[other_id]["name"],
                "status": rows[other_id]["status"],
                "instance_count": rows[other_id]["instance_count"],
                "similarity": face_clustering.cosine_similarity(target, centroid),
            }
        )
    return sorted(scored, key=lambda c: -c["similarity"])[:limit]


def merge_face_clusters(conn: sqlite3.Connection, cluster_ids: list[int]) -> int:
    """Merges 2+ face clusters into one and returns the survivor's id.

    Survivor choice follows §4.3: prefer an already-named cluster, then the
    one with the most instances — keeping the name the user already gave,
    and minimising the number of rows that have to move.

    Moved instances are marked is_manual_override, so a future re-clustering
    pass won't quietly undo a merge the user asked for (§4.4).
    """
    if len(cluster_ids) < 2:
        raise ValueError("Merging needs at least two clusters.")

    rows = conn.execute(
        f"""
        SELECT fc.id, fc.name, fc.status, COUNT(fi.id) AS instance_count
        FROM face_cluster fc
        LEFT JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
        WHERE fc.id IN ({','.join('?' * len(cluster_ids))})
        GROUP BY fc.id
        """,
        cluster_ids,
    ).fetchall()
    if len(rows) < 2:
        raise ValueError("Some clusters to merge do not exist.")

    survivor = max(rows, key=lambda r: (r["status"] == "named", r["instance_count"]))
    survivor_id = survivor["id"]
    absorbed = [r["id"] for r in rows if r["id"] != survivor_id]

    conn.execute(
        f"""
        UPDATE face_instance SET face_cluster_id = ?, is_manual_override = 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE face_cluster_id IN ({','.join('?' * len(absorbed))})
        """,
        [survivor_id, *absorbed],
    )
    conn.execute(
        f"DELETE FROM face_cluster WHERE id IN ({','.join('?' * len(absorbed))})", absorbed
    )
    _log_cluster_event(
        conn,
        cluster_type="face",
        event_type="merge",
        source_ids=sorted(cluster_ids),
        resulting_ids=[survivor_id],
    )
    return survivor_id


def merge_location_clusters(conn: sqlite3.Connection, cluster_ids: list[int]) -> int:
    """Location equivalent. Also recomputes the survivor's centroid from
    all photos it now holds, since a stale centroid would misplace it on
    the map and skew future assignment."""
    if len(cluster_ids) < 2:
        raise ValueError("Merging needs at least two clusters.")

    rows = conn.execute(
        f"""
        SELECT lc.id, lc.name, lc.status, COUNT(pl.photo_id) AS instance_count
        FROM location_cluster lc
        LEFT JOIN photo_location pl ON pl.location_cluster_id = lc.id
        WHERE lc.id IN ({','.join('?' * len(cluster_ids))})
        GROUP BY lc.id
        """,
        cluster_ids,
    ).fetchall()
    if len(rows) < 2:
        raise ValueError("Some clusters to merge do not exist.")

    survivor = max(rows, key=lambda r: (r["status"] == "named", r["instance_count"]))
    survivor_id = survivor["id"]
    absorbed = [r["id"] for r in rows if r["id"] != survivor_id]

    conn.execute(
        f"""
        UPDATE photo_location SET location_cluster_id = ?, is_manual_override = 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE location_cluster_id IN ({','.join('?' * len(absorbed))})
        """,
        [survivor_id, *absorbed],
    )
    conn.execute(
        f"DELETE FROM location_cluster WHERE id IN ({','.join('?' * len(absorbed))})", absorbed
    )

    centre = conn.execute(
        """
        SELECT AVG(COALESCE(p.gps_override_lat, p.gps_lat)) AS lat,
               AVG(COALESCE(p.gps_override_lng, p.gps_lng)) AS lng
        FROM photo_location pl JOIN photo p ON p.id = pl.photo_id
        WHERE pl.location_cluster_id = ?
        """,
        (survivor_id,),
    ).fetchone()
    if centre and centre["lat"] is not None:
        update_location_cluster_centroid(conn, survivor_id, centre["lat"], centre["lng"])

    _log_cluster_event(
        conn,
        cluster_type="location",
        event_type="merge",
        source_ids=sorted(cluster_ids),
        resulting_ids=[survivor_id],
    )
    return survivor_id


def split_face_cluster(
    conn: sqlite3.Connection, cluster_id: int, face_instance_ids: list[int]
) -> int:
    """Moves the given faces out of `cluster_id` into a new unlabeled
    cluster, returning its id (§4.3).

    Refuses to move *every* face, which would leave an empty husk behind
    and just rename the cluster rather than split it.
    """
    if not face_instance_ids:
        raise ValueError("Splitting needs at least one face.")

    total = conn.execute(
        "SELECT COUNT(*) AS c FROM face_instance "
        "WHERE face_cluster_id = ? AND false_positive_at IS NULL",
        (cluster_id,),
    ).fetchone()["c"]
    if len(face_instance_ids) >= total:
        raise ValueError("Cannot split every face out of a cluster — nothing would remain.")

    representative = conn.execute(
        "SELECT photo_id FROM face_instance WHERE id = ?", (face_instance_ids[0],)
    ).fetchone()
    new_cluster_id = insert_face_cluster(
        conn, representative_photo_id=representative["photo_id"] if representative else None
    )

    conn.execute(
        f"""
        UPDATE face_instance SET face_cluster_id = ?, is_manual_override = 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id IN ({','.join('?' * len(face_instance_ids))}) AND face_cluster_id = ?
        """,
        [new_cluster_id, *face_instance_ids, cluster_id],
    )
    _log_cluster_event(
        conn,
        cluster_type="face",
        event_type="split",
        source_ids=[cluster_id],
        resulting_ids=[cluster_id, new_cluster_id],
    )
    return new_cluster_id


def set_location_geocoded_name(conn: sqlite3.Connection, cluster_id: int, name: Optional[str]) -> None:
    """Stores the reverse-geocoded suggestion (§3 step 6). Records the
    timestamp even when the provider had nothing, so a cluster with no
    answer isn't retried on every run."""
    conn.execute(
        """
        UPDATE location_cluster
        SET geocoded_name = ?, geocoded_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (name, cluster_id),
    )


def set_location_ocr_name(conn: sqlite3.Connection, cluster_id: int, encoded: Optional[str]) -> None:
    """Stores OCR candidates read off the cluster's photos, as JSON in a TEXT
    column (see ocr.encode_candidates)."""
    conn.execute(
        """
        UPDATE location_cluster
        SET ocr_name = ?, ocr_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (encoded, cluster_id),
    )


def location_clusters_needing_suggestions(
    conn: sqlite3.Connection, *, source: str
) -> list[dict]:
    """Clusters that haven't had the given suggestion pass run yet."""
    column = {"geocode": "geocoded_at", "ocr": "ocr_at"}[source]
    return [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT id, centroid_lat, centroid_lng
            FROM location_cluster
            WHERE {column} IS NULL
            ORDER BY id
            """
        )
    ]


def exclude_cluster(conn: sqlite3.Connection, kind: str, cluster_id: int) -> None:
    """Hides a cluster from the labeling queue, albums, uncategorized lists,
    and merge candidates — for strangers caught in the background of a shot.

    Orthogonal to `status`, so an excluded cluster keeps whatever labeling
    state it had and can be restored unchanged. Nothing is deleted: the
    faces stay in the database, they simply stop being offered.
    """
    conn.execute(
        f"""
        UPDATE {_table_for(kind)}
        SET excluded_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (cluster_id,),
    )


def restore_cluster(conn: sqlite3.Connection, kind: str, cluster_id: int) -> None:
    conn.execute(
        f"""
        UPDATE {_table_for(kind)}
        SET excluded_at = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (cluster_id,),
    )


def load_excluded(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Excluded clusters, most recently hidden first — so a mistaken
    exclusion is easy to find and undo."""
    faces = conn.execute(
        """
        SELECT fc.id, fc.name, fc.status, fc.excluded_at, COUNT(fi.id) AS instance_count
        FROM face_cluster fc
        JOIN face_instance fi ON fi.face_cluster_id = fc.id AND fi.false_positive_at IS NULL
        WHERE fc.excluded_at IS NOT NULL
        GROUP BY fc.id ORDER BY fc.excluded_at DESC
        """
    ).fetchall()
    places = conn.execute(
        """
        SELECT lc.id, lc.name, lc.status, lc.excluded_at, COUNT(pl.photo_id) AS instance_count
        FROM location_cluster lc
        JOIN photo_location pl ON pl.location_cluster_id = lc.id
        WHERE lc.excluded_at IS NOT NULL
        GROUP BY lc.id ORDER BY lc.excluded_at DESC
        """
    ).fetchall()
    return {"people": [dict(r) for r in faces], "places": [dict(r) for r in places]}


def excluded_count(conn: sqlite3.Connection) -> int:
    return (
        conn.execute("SELECT COUNT(*) AS c FROM face_cluster WHERE excluded_at IS NOT NULL").fetchone()["c"]
        + conn.execute("SELECT COUNT(*) AS c FROM location_cluster WHERE excluded_at IS NOT NULL").fetchone()["c"]
    )


def clusters_sharing_a_name(conn: sqlite3.Connection, kind: str, name: str) -> list[dict]:
    """Every cluster of this kind carrying `name`, largest first.

    Two clusters with the same user-assigned name are the strongest merge
    signal available — stronger than embedding similarity, because the user
    has stated they are the same person rather than the model guessing it.
    """
    table = _table_for(kind)
    if kind == "face":
        sql = f"""
            SELECT c.id, COUNT(fi.id) AS instance_count
            FROM {table} c LEFT JOIN face_instance fi
                ON fi.face_cluster_id = c.id AND fi.false_positive_at IS NULL
            WHERE c.name = ? AND c.status = 'named'
            GROUP BY c.id ORDER BY instance_count DESC
        """
    else:
        sql = f"""
            SELECT c.id, COUNT(pl.photo_id) AS instance_count
            FROM {table} c LEFT JOIN photo_location pl ON pl.location_cluster_id = c.id
            WHERE c.name = ? AND c.status = 'named'
            GROUP BY c.id ORDER BY instance_count DESC
        """
    return [dict(r) for r in conn.execute(sql, (name,))]


def duplicate_named_clusters(conn: sqlite3.Connection) -> list[dict]:
    """Names applied to more than one cluster — i.e. people or places the
    user has effectively already said are the same thing, but which the
    clustering still holds apart. Surfaced so they can be merged in one go."""
    duplicates = []
    for kind, grouped in (("face", load_albums(conn)["people"]), ("place", load_albums(conn)["places"])):
        for album in grouped:
            if len(album["ids"]) > 1:
                duplicates.append(
                    {
                        "kind": kind,
                        "name": album["name"],
                        "ids": album["ids"],
                        "instance_count": album["instance_count"],
                    }
                )
    return sorted(duplicates, key=lambda d: -d["instance_count"])


def cluster_events(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    return [
        {
            "id": r["id"],
            "cluster_type": r["cluster_type"],
            "event_type": r["event_type"],
            "source_cluster_ids": json.loads(r["source_cluster_ids"]),
            "resulting_cluster_ids": json.loads(r["resulting_cluster_ids"]),
            "timestamp": r["timestamp"],
        }
        for r in conn.execute(
            "SELECT * FROM cluster_event ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for the tables ingestion touches — used by the CLI to
    report what a run actually produced."""
    tables = ("photo", "face_cluster", "face_instance", "location_cluster", "photo_location")
    return {
        table: conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        for table in tables
    }


# ---------------------------------------------------------------------
# Events — new, not in the original spec. Purely user-created: the user
# selects a batch of photos and names the occasion directly, unlike
# FaceCluster/LocationCluster which a detector proposes and the user only
# labels. Many-to-many with photos (photo_event), unlike PhotoLocation's
# one-per-photo shape — an outing can also be "someone's birthday".
# ---------------------------------------------------------------------


def create_event(conn: sqlite3.Connection, name: str, description: Optional[str] = None) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Event name cannot be empty.")
    cursor = conn.execute(
        "INSERT INTO event (name, description) VALUES (?, ?)", (name, description)
    )
    return cursor.lastrowid


def add_photos_to_event(conn: sqlite3.Connection, event_id: int, photo_ids: list[int]) -> int:
    """Idempotent — re-adding a photo already in the event is a no-op
    (INSERT OR IGNORE against the (photo_id, event_id) primary key), so
    the bulk-select UI doesn't need to track what's already there."""
    if not photo_ids:
        return 0
    before = conn.execute(
        "SELECT COUNT(*) AS c FROM photo_event WHERE event_id = ?", (event_id,)
    ).fetchone()["c"]
    conn.executemany(
        "INSERT OR IGNORE INTO photo_event (photo_id, event_id) VALUES (?, ?)",
        [(pid, event_id) for pid in photo_ids],
    )
    after = conn.execute(
        "SELECT COUNT(*) AS c FROM photo_event WHERE event_id = ?", (event_id,)
    ).fetchone()["c"]
    return after - before


def remove_photo_from_event(conn: sqlite3.Connection, event_id: int, photo_id: int) -> None:
    conn.execute(
        "DELETE FROM photo_event WHERE event_id = ? AND photo_id = ?", (event_id, photo_id)
    )


def find_event_by_name(conn: sqlite3.Connection, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM event WHERE name = ?", (name.strip(),)).fetchone()
    return row["id"] if row else None


def get_or_create_event(conn: sqlite3.Connection, name: str) -> int:
    """The same 'type a name, match-or-create' pattern used for naming
    faces and places — bulk-assigning a batch of photos to an event should
    feel like the same action as naming a person, not a different UI idiom."""
    existing = find_event_by_name(conn, name)
    return existing if existing is not None else create_event(conn, name)


def rename_event(conn: sqlite3.Connection, event_id: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Event name cannot be empty.")
    conn.execute(
        "UPDATE event SET name = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (name, event_id),
    )


def set_event_description(conn: sqlite3.Connection, event_id: int, description: str) -> None:
    conn.execute(
        "UPDATE event SET description = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (description.strip() or None, event_id),
    )


def set_event_autobio_excluded(conn: sqlite3.Connection, event_id: int, excluded: bool) -> None:
    """Whether photos tagged with this event are dropped from Diary/Autobio
    generation entirely — see photos_for_date() and migration
    0015_event_autobio_exclude.sql."""
    conn.execute(
        "UPDATE event SET excluded_from_autobio = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (1 if excluded else 0, event_id),
    )


def delete_event(conn: sqlite3.Connection, event_id: int) -> None:
    """Events are pure user data with nothing to preserve for undo the way
    a detected face cluster has (there's no re-detection to protect against)
    — a straightforward delete, not a hide-flag."""
    conn.execute("DELETE FROM event WHERE id = ?", (event_id,))


def list_events(conn: sqlite3.Connection) -> list[dict]:
    """Every event with its photo count, most photos first — mirrors how
    albums are ordered (§4.1)."""
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT e.id, e.name, e.description, e.excluded_from_autobio,
                   COUNT(pe.photo_id) AS instance_count
            FROM event e LEFT JOIN photo_event pe ON pe.event_id = e.id
            GROUP BY e.id
            ORDER BY instance_count DESC, e.name
            """
        )
    ]


def event_detail(conn: sqlite3.Connection, event_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM event WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        return None
    photos = [
        {
            "photo_id": r["id"],
            "path": r["original_storage_path"],
            "filename": r["original_filename"],
            "taken_at": r["effective_taken_at"],
        }
        for r in conn.execute(
            """
            SELECT p.id, p.original_storage_path, p.original_filename,
                   COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at
            FROM photo_event pe JOIN photo p ON p.id = pe.photo_id
            WHERE pe.event_id = ?
            ORDER BY effective_taken_at
            """,
            (event_id,),
        )
    ]
    return {"event": dict(row), "photos": photos}


def events_for_photo(conn: sqlite3.Connection, photo_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT e.id, e.name FROM event e
            JOIN photo_event pe ON pe.event_id = e.id
            WHERE pe.photo_id = ? ORDER BY e.name
            """,
            (photo_id,),
        )
    ]


# ---------------------------------------------------------------------
# Chronological browsing ("photo by time") + descriptions — new
# ---------------------------------------------------------------------


def list_all_photos(conn: sqlite3.Connection, *, limit: int = 200, offset: int = 0) -> list[dict]:
    """Every photo, newest first — the home page's time-ordered view,
    independent of whether a photo has been sorted into any person/place/
    event yet. Paginated: a real library runs to hundreds/thousands of
    photos, and this is meant for scrolling through recent imports, not
    rendering the whole library in one page load.

    Includes each photo's place name (if its location has been labeled) so
    the timeline can show it in a per-day header, the way Google Photos'
    own timeline does — a LEFT JOIN since most photos won't have a named
    place yet, and that's fine, not an error."""
    return [
        {
            "photo_id": r["id"],
            "taken_at": r["effective_taken_at"],
            "place_name": r["place_name"],
        }
        for r in conn.execute(
            """
            SELECT p.id AS id,
                   COALESCE(p.taken_at_override, p.taken_at) AS effective_taken_at,
                   lc.name AS place_name
            FROM photo p
            LEFT JOIN photo_location pl ON pl.photo_id = p.id
            LEFT JOIN location_cluster lc ON lc.id = pl.location_cluster_id
            ORDER BY effective_taken_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
    ]


def total_photo_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM photo").fetchone()["c"]


def set_cluster_description(conn: sqlite3.Connection, kind: str, cluster_id: int, description: str) -> None:
    conn.execute(
        f"""
        UPDATE {_table_for(kind)}
        SET description = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (description.strip() or None, cluster_id),
    )


def set_photo_description(conn: sqlite3.Connection, photo_id: int, description: str) -> None:
    conn.execute(
        """
        UPDATE photo SET description = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE id = ?
        """,
        (description.strip() or None, photo_id),
    )


# ---------------------------------------------------------------------
# Pending Google Photos Picker session — persisted server-side so the
# import can be resumed from any tab, not just the one that started it
# (see migration 0009 for why).
# ---------------------------------------------------------------------


def save_pending_import_session(conn: sqlite3.Connection, *, session_id: str, picker_uri: str) -> None:
    """Replaces any previously pending session — only one is ever
    meaningful at a time in this single-user app, and an old, likely-dead
    session isn't worth keeping around once a new one starts."""
    conn.execute(
        """
        INSERT INTO pending_import_session (id, session_id, picker_uri)
        VALUES (1, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            session_id = excluded.session_id,
            picker_uri = excluded.picker_uri,
            created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (session_id, picker_uri),
    )


def get_pending_import_session(conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT session_id, picker_uri, created_at FROM pending_import_session WHERE id = 1"
    ).fetchone()
    return dict(row) if row else None


def claim_pending_import_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Atomically claims a pending session for actual ingestion — deletes
    the row only if it still exists and still matches this session_id, and
    reports whether *this* call was the one that got it.

    Exists because the picking page (dedicated tab, auto-polling) and the
    /import hub's "Finish that import" banner are now two independent live
    paths that can both reach /import/continue for the very same session —
    observed in practice as a real crash: the loser's `list_media_items()`
    404s because the winner had already fetched the items and deleted the
    session on Google's side out from under it. Only the winner (this
    returns True) should proceed to fetch/ingest; a loser (False) should
    treat it as already handled, not retry the same work."""
    cursor = conn.execute(
        "DELETE FROM pending_import_session WHERE id = 1 AND session_id = ?",
        (session_id,),
    )
    return cursor.rowcount > 0


def clear_pending_import_session(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM pending_import_session WHERE id = 1")


# ---------------------------------------------------------------------
# Autobio (§4.6) — daily narrative drafting/editing.
# ---------------------------------------------------------------------


def photos_for_date(conn: sqlite3.Connection, date: str) -> list[dict]:
    """Every photo taken on `date` ("YYYY-MM-DD"), oldest first
    (chronological narrative order), each with the named people/places
    and Event labels actually attached to *that* photo — the LLM prompt
    wants per-moment detail, not just the day's aggregate, and a later
    per-segment correction needs the same per-photo detail to resolve
    back to a photo. An Event label (§4.1's user-created grouping, e.g.
    "원필 생일") is often the actual *reason* a day's photos exist — worth
    giving the model directly rather than leaving it to guess.

    Matched against taken_at as already stored — naive local EXIF time
    (see README's "Known limitation — timezones" note) — which is exactly
    the "what did I do on this day" grouping wants, no conversion needed.
    """
    photos = [
        {
            "photo_id": r["id"],
            "taken_at": r["effective_taken_at"],
            "description": r["description"],
            "people": [],
            "places": [],
            "events": [],
        }
        for r in conn.execute(
            """
            SELECT id, COALESCE(taken_at_override, taken_at) AS effective_taken_at, description
            FROM photo
            WHERE date(COALESCE(taken_at_override, taken_at)) = ?
            ORDER BY effective_taken_at
            """,
            (date,),
        )
    ]
    by_id = {p["photo_id"]: p for p in photos}
    if not by_id:
        return []

    placeholders = ",".join("?" * len(by_id))
    ids = list(by_id.keys())

    for r in conn.execute(
        f"""
        SELECT DISTINCT fi.photo_id, fc.name
        FROM face_instance fi
        JOIN face_cluster fc ON fc.id = fi.face_cluster_id
        WHERE fi.photo_id IN ({placeholders}) AND fc.status = 'named' AND fi.false_positive_at IS NULL
        """,
        ids,
    ):
        by_id[r["photo_id"]]["people"].append(r["name"])

    for r in conn.execute(
        f"""
        SELECT pl.photo_id, lc.name
        FROM photo_location pl
        JOIN location_cluster lc ON lc.id = pl.location_cluster_id
        WHERE pl.photo_id IN ({placeholders}) AND lc.status = 'named'
        """,
        ids,
    ):
        by_id[r["photo_id"]]["places"].append(r["name"])

    # A photo tagged with an excluded event is dropped from the day
    # entirely, not just its event name hidden — someone excluding a group
    # from Diary/Autobio wants those photos out of the AI prompt, not
    # mentioned-minus-the-label. If a photo belongs to more than one event
    # and only some are excluded, excluding wins: the whole point is "never
    # let this show up in generated text."
    excluded_photo_ids = set()
    for r in conn.execute(
        f"""
        SELECT pe.photo_id, e.name, e.excluded_from_autobio
        FROM photo_event pe
        JOIN event e ON e.id = pe.event_id
        WHERE pe.photo_id IN ({placeholders})
        """,
        ids,
    ):
        if r["excluded_from_autobio"]:
            excluded_photo_ids.add(r["photo_id"])
        else:
            by_id[r["photo_id"]]["events"].append(r["name"])

    if excluded_photo_ids:
        photos = [p for p in photos if p["photo_id"] not in excluded_photo_ids]

    return photos


def count_unlabeled_for_date(conn: sqlite3.Connection, date: str) -> int:
    """How many distinct unlabeled/not_sure face clusters or unlabeled
    place clusters appear in this day's photos — backs §4.6's
    "N unlabeled people appear today" nudge. Excluded (hidden) clusters
    don't count — the user already dismissed those as not worth labeling."""
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(DISTINCT fc.id)
           FROM face_instance fi
           JOIN face_cluster fc ON fc.id = fi.face_cluster_id
           JOIN photo p ON p.id = fi.photo_id
           WHERE date(COALESCE(p.taken_at_override, p.taken_at)) = ?
             AND fc.status != 'named' AND fc.excluded_at IS NULL
             AND fi.false_positive_at IS NULL)
          +
          (SELECT COUNT(DISTINCT lc.id)
           FROM photo_location pl
           JOIN location_cluster lc ON lc.id = pl.location_cluster_id
           JOIN photo p ON p.id = pl.photo_id
           WHERE date(COALESCE(p.taken_at_override, p.taken_at)) = ?
             AND lc.status != 'named' AND lc.excluded_at IS NULL)
          AS total
        """,
        (date, date),
    ).fetchone()
    return row["total"] or 0


def _autobio_entry_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "date": row["date"],
        "segments": json.loads(row["segments"]),
        "draft_text": row["draft_text"],
        # final_text is NOT NULL and defaults to draft_text itself (set
        # equal to it whenever a fresh draft lands with nothing to
        # protect) — so it's always the text to actually show; there's no
        # separate "effective_text" to compute.
        "final_text": row["final_text"],
        "is_edited": row["final_text"] != row["draft_text"],
        "has_unlabeled": bool(row["has_unlabeled_flag"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_autobio_draft(
    conn: sqlite3.Connection, *, date: str, segments: list[dict], draft_text: str, has_unlabeled: bool
) -> int:
    """Creates the entry for `date`, or replaces its draft if one already
    exists. final_text is only overwritten to match the new draft when it
    hadn't diverged from the *previous* draft — i.e. there was no user
    edit to protect. If the user had already edited it, final_text is left
    exactly as they left it: regenerating must never silently discard a
    correction someone already made. Returns the entry id."""
    existing = conn.execute(
        "SELECT id, draft_text, final_text FROM autobio_entry WHERE date = ?", (date,)
    ).fetchone()

    if existing:
        was_edited = existing["final_text"] != existing["draft_text"]
        if was_edited:
            conn.execute(
                """
                UPDATE autobio_entry
                SET segments = ?, draft_text = ?, has_unlabeled_flag = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE date = ?
                """,
                (json.dumps(segments), draft_text, int(has_unlabeled), date),
            )
        else:
            conn.execute(
                """
                UPDATE autobio_entry
                SET segments = ?, draft_text = ?, final_text = ?, has_unlabeled_flag = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE date = ?
                """,
                (json.dumps(segments), draft_text, draft_text, int(has_unlabeled), date),
            )
        return existing["id"]

    cursor = conn.execute(
        """
        INSERT INTO autobio_entry (date, segments, draft_text, final_text, has_unlabeled_flag)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date, json.dumps(segments), draft_text, draft_text, int(has_unlabeled)),
    )
    return cursor.lastrowid


def get_autobio_entry(conn: sqlite3.Connection, date: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM autobio_entry WHERE date = ?", (date,)).fetchone()
    return _autobio_entry_dict(row) if row else None


def set_autobio_final_text(conn: sqlite3.Connection, date: str, final_text: str) -> None:
    """Blunt whole-entry override — sets final_text directly without
    touching segments, so it can drift out of sync with the per-segment
    editor (set_autobio_segment_text) if both are used on the same entry.
    The web UI no longer exposes this as the primary editing path (see
    autobio_entry.html — per-segment editing is), but it's kept as a
    lower-level operation other callers (a script, a future API) might
    still reasonably want."""
    conn.execute(
        """
        UPDATE autobio_entry
        SET final_text = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE date = ?
        """,
        (final_text, date),
    )


def set_autobio_segment_text(
    conn: sqlite3.Connection, date: str, index: int, text: str, *, edited: bool = True
) -> None:
    """§4.6's per-segment correction flow: updates just one segment's
    text (marked edited by default — pass edited=False for a fresh LLM
    regeneration, see autobio.regenerate_segment, which isn't a user
    edit), then reassembles final_text from every segment's *current*
    text so the whole-entry view/export always reflects the latest
    per-segment state."""
    entry = get_autobio_entry(conn, date)
    if entry is None:
        raise ValueError(f"No autobio entry for {date}.")
    segments = entry["segments"]
    if not (0 <= index < len(segments)):
        raise IndexError(f"No segment {index} for {date} (has {len(segments)}).")

    segments[index]["text"] = text
    segments[index]["edited"] = edited
    final_text = "\n\n".join(s["text"] for s in segments)

    conn.execute(
        """
        UPDATE autobio_entry
        SET segments = ?, final_text = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE date = ?
        """,
        (json.dumps(segments), final_text, date),
    )


def list_autobio_entries(conn: sqlite3.Connection) -> list[dict]:
    """Every generated day, newest first — the history view."""
    return [
        _autobio_entry_dict(r)
        for r in conn.execute("SELECT * FROM autobio_entry ORDER BY date DESC")
    ]


def delete_autobio_entry(conn: sqlite3.Connection, date: str) -> None:
    """Removes a generated Diary entry entirely — a fresh "Generate
    narrative" for the same date starts over from scratch, same as if it
    had never been drafted. Doesn't touch the underlying photos or their
    labels; only the drafted/edited text. Any combined narrative already
    built from this date is untouched too — autobio_summary stores its
    own text snapshot, not a live reference back to autobio_entry."""
    conn.execute("DELETE FROM autobio_entry WHERE date = ?", (date,))


# ---------------------------------------------------------------------
# Autobio combined narrative (§4.6 "Combined narrative") — a date-range
# summary built from already-generated/edited daily entries.
#
# Unlike autobio_entry, this table (from the original schema, migration
# 0001) has a single narrative_text column — no draft/final split — so
# there's no way to protect a prior edit the way save_autobio_draft()
# does for daily entries: regenerating a summary always overwrites
# narrative_text. The web layer confirms before regenerating for exactly
# this reason.
# ---------------------------------------------------------------------


def _autobio_summary_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "source_entry_ids": json.loads(row["source_entry_ids"]),
        "text": row["narrative_text"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_autobio_summary(
    conn: sqlite3.Connection, *, start_date: str, end_date: str, source_entry_ids: list[int], text: str
) -> int:
    """Creates the summary for (start_date, end_date), or replaces it if
    one already exists for that exact range — (start_date, end_date) is
    treated as a natural key, matching how a single date works for
    autobio_entry, rather than accumulating a new row per regeneration."""
    existing = conn.execute(
        "SELECT id FROM autobio_summary WHERE start_date = ? AND end_date = ?", (start_date, end_date)
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE autobio_summary
            SET source_entry_ids = ?, narrative_text = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (json.dumps(source_entry_ids), text, existing["id"]),
        )
        return existing["id"]
    cursor = conn.execute(
        """
        INSERT INTO autobio_summary (start_date, end_date, source_entry_ids, narrative_text)
        VALUES (?, ?, ?, ?)
        """,
        (start_date, end_date, json.dumps(source_entry_ids), text),
    )
    return cursor.lastrowid


def get_autobio_summary(conn: sqlite3.Connection, start_date: str, end_date: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM autobio_summary WHERE start_date = ? AND end_date = ?", (start_date, end_date)
    ).fetchone()
    return _autobio_summary_dict(row) if row else None


def set_autobio_summary_text(conn: sqlite3.Connection, start_date: str, end_date: str, text: str) -> None:
    conn.execute(
        """
        UPDATE autobio_summary
        SET narrative_text = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE start_date = ? AND end_date = ?
        """,
        (text, start_date, end_date),
    )


def list_autobio_summaries(conn: sqlite3.Connection) -> list[dict]:
    """Every generated combined narrative, most recent range first."""
    return [
        _autobio_summary_dict(r)
        for r in conn.execute("SELECT * FROM autobio_summary ORDER BY start_date DESC")
    ]


def delete_autobio_summary(conn: sqlite3.Connection, start_date: str, end_date: str) -> None:
    """Removes a combined narrative — the daily Diary entries it was built
    from are untouched (see delete_autobio_entry's docstring; the
    reference is one-directional)."""
    conn.execute(
        "DELETE FROM autobio_summary WHERE start_date = ? AND end_date = ?", (start_date, end_date)
    )


def autobio_entry_dates_for_ids(conn: sqlite3.Connection, entry_ids: list[int]) -> list[str]:
    """Resolves autobio_entry.id values (as stored in a summary's
    source_entry_ids) back to their dates, chronological order — for
    linking a combined narrative to the daily entries it was built from."""
    if not entry_ids:
        return []
    placeholders = ",".join("?" * len(entry_ids))
    rows = conn.execute(
        f"SELECT date FROM autobio_entry WHERE id IN ({placeholders}) ORDER BY date", entry_ids
    ).fetchall()
    return [r["date"] for r in rows]


def get_autobio_settings(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT show_unlabeled_nudge FROM autobio_settings WHERE id = 1").fetchone()
    return {"show_unlabeled_nudge": bool(row["show_unlabeled_nudge"]) if row else True}


def set_autobio_show_unlabeled_nudge(conn: sqlite3.Connection, enabled: bool) -> None:
    conn.execute("UPDATE autobio_settings SET show_unlabeled_nudge = ? WHERE id = 1", (int(enabled),))


# ---------------------------------------------------------------------
# General app settings: speech-typing language (§4.2), narrative language
# (Autobio/Diary), and UI language.
# ---------------------------------------------------------------------

# BCP-47 tags the Web Speech API expects — matches this app's existing
# regional focus (Korea/Japan, see geocoding.py) plus English as a safe
# default. A code not in this list is still accepted by set_speech_language
# (so a future language doesn't need a migration to use), just not offered
# as a dropdown option.
SPEECH_LANGUAGES = [
    ("ko-KR", "한국어 (Korean)"),
    ("en-US", "English (US)"),
    ("ja-JP", "日本語 (Japanese)"),
]

# Plain two-letter codes (not BCP-47 — these aren't fed to a browser API,
# just used as dict keys for i18n.py and as a plain instruction to the
# LLM), shared by both narrative_language and ui_language since they offer
# the same six languages.
LANGUAGES = [
    ("en", "English"),
    ("ko", "한국어 (Korean)"),
    ("ja", "日本語 (Japanese)"),
    ("uk", "Українська (Ukrainian)"),
    ("es", "Español (Spanish)"),
    ("fr", "Français (French)"),
]


def get_app_settings(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT speech_language, narrative_language, ui_language, llm_provider, originals_dir FROM app_settings WHERE id = 1"
    ).fetchone()
    if not row:
        return {
            "speech_language": "ko-KR",
            "narrative_language": "ko",
            "ui_language": "ko",
            "llm_provider": "",
            "originals_dir": "",
        }
    return dict(row)


def set_llm_provider(conn: sqlite3.Connection, provider: str) -> None:
    """`provider` is 'anthropic', 'openai', or '' for auto-detect (see
    llm.resolve_provider_and_key — this is the Settings-page equivalent of
    its AUTOBIO_LLM_PROVIDER env var override, which still takes priority
    if set)."""
    conn.execute("UPDATE app_settings SET llm_provider = ? WHERE id = 1", (provider,))


def set_originals_dir(conn: sqlite3.Connection, path: str) -> None:
    """`path` is an absolute filesystem path, or '' to fall back to
    web.py's DEFAULT_ORIGINALS_DIR. Only affects where future imports
    save their originals — see this migration's docstring
    (0014_originals_dir.sql)."""
    conn.execute("UPDATE app_settings SET originals_dir = ? WHERE id = 1", (path,))


def set_speech_language(conn: sqlite3.Connection, language: str) -> None:
    conn.execute("UPDATE app_settings SET speech_language = ? WHERE id = 1", (language,))


def set_narrative_language(conn: sqlite3.Connection, language: str) -> None:
    conn.execute("UPDATE app_settings SET narrative_language = ? WHERE id = 1", (language,))


def set_ui_language(conn: sqlite3.Connection, language: str) -> None:
    conn.execute("UPDATE app_settings SET ui_language = ? WHERE id = 1", (language,))

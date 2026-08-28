"""The core ingestion pipeline — §3, wired to the database (§6 step 7).

Assembles the four pieces validated separately by the spike scripts:
Picker fetch + storage, face detection/clustering, Takeout GPS backfill,
and location clustering.

Two places where this deliberately departs from §3 as written, both
because real data forced it (see README):

  * §3 step 3 has ingestion read GPS from the processing fetch's EXIF.
    The Picker API strips GPS from every fetch variant, so there is no GPS
    to read at ingestion time. Photos land with gps_lat/gps_lng NULL and a
    separate Takeout backfill pass fills them in — hence
    `backfill_gps_from_takeout()` below, and hence location clustering
    running as its own pass rather than inline per photo.

  * §3 step 3 also reads taken_at from that EXIF. That part still works on
    the '=d' original (only GPS is stripped), so taken_at is read from the
    saved original rather than from the discarded processing bytes.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Protocol

from . import exif, face_clustering, geocoding, location_clustering, repository, takeout


class _PickerClient(Protocol):
    """The picker_client surface ingestion uses — declared so tests can
    substitute a fake without a live Google session."""

    def fetch_processing_bytes(self, access_token: str, base_url: str) -> bytes: ...
    def fetch_original_bytes(self, access_token: str, base_url: str) -> bytes: ...


class _FaceDetector(Protocol):
    def __call__(self, image_bytes: bytes) -> list[tuple[dict, list[float]]]:
        """Returns (bounding_box, embedding) per detected face."""


@dataclass
class IngestionResult:
    imported: int = 0
    skipped_already_imported: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (filename, error)
    faces_detected: int = 0
    face_clusters_created: int = 0

    @property
    def attempted(self) -> int:
        return self.imported + self.skipped_already_imported + len(self.failed)


def ingest_picked_items(
    conn: sqlite3.Connection,
    items: list[dict],
    *,
    access_token: str,
    picker_client: _PickerClient,
    storage_adapter,
    detect_faces: Optional[_FaceDetector] = None,
    face_similarity_threshold: float = face_clustering.DEFAULT_SIMILARITY_THRESHOLD,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> IngestionResult:
    """Runs §3 steps 1-5 and 9 for each picked item.

    Each photo is committed in its own transaction: one bad photo (a
    transient fetch failure, an unreadable image) leaves the rest of the
    batch intact and can be retried by re-running, since already-imported
    photos are skipped on picker_media_id.
    """
    result = IngestionResult()

    centroids = [
        face_clustering.FaceClusterCentroid(cluster_id=cid, centroid=centroid, count=count)
        for cid, centroid, count in repository.load_face_cluster_centroids(conn)
    ]
    # §3 step 5: checked once per new cluster, not stored per-instance —
    # see face_clustering.find_seed_match for why this only ever produces a
    # suggestion, never an auto-assigned name.
    seed_faces = repository.load_seed_faces(conn) if detect_faces else []

    for item in items:
        media_file = item["mediaFile"]
        filename = media_file.get("filename") or item["id"]

        if repository.photo_id_for_picker_media_id(conn, item["id"]) is not None:
            result.skipped_already_imported += 1
            if on_progress:
                on_progress(filename, "already imported, skipped")
            continue

        try:
            _ingest_one(
                conn,
                item,
                filename=filename,
                access_token=access_token,
                picker_client=picker_client,
                storage_adapter=storage_adapter,
                detect_faces=detect_faces,
                centroids=centroids,
                seed_faces=seed_faces,
                face_similarity_threshold=face_similarity_threshold,
                result=result,
            )
        except Exception as e:  # noqa: BLE001 - per-photo guard, mirrors picker_spike.py
            conn.rollback()
            result.failed.append((filename, str(e)))
            if on_progress:
                on_progress(filename, f"FAILED: {e}")
            continue

        result.imported += 1
        if on_progress:
            on_progress(filename, "imported")

    return result


def _ingest_one(
    conn: sqlite3.Connection,
    item: dict,
    *,
    filename: str,
    access_token: str,
    picker_client: _PickerClient,
    storage_adapter,
    detect_faces: Optional[_FaceDetector],
    centroids: list[face_clustering.FaceClusterCentroid],
    seed_faces: list[tuple[int, str, list[float]]],
    face_similarity_threshold: float,
    result: IngestionResult,
) -> None:
    base_url = item["mediaFile"]["baseUrl"]

    # §3 step 5: the original must be fetched now, while baseUrl is valid.
    #
    # This is the only fetch. §3 step 2 specifies a second, smaller fetch
    # for face detection, but that predates §5's decision to always retain
    # the full original — given the original is downloaded regardless,
    # fetching a downscaled copy of the same photo is redundant, and it
    # doubled the per-photo API calls that produced 429 rate-limiting in
    # practice. Detecting on the original also keeps face sizes at the
    # resolution the distance threshold was tuned against; see
    # face_embeddings.detect_faces_in_bytes.
    original_bytes = picker_client.fetch_original_bytes(access_token, base_url)

    # §3 step 3: taken_at survives on the original ('=d' strips only GPS).
    # Falling back to import time keeps a photo with unreadable EXIF
    # ingestible rather than failing the row — taken_at is NOT NULL, and
    # §4.4 lets the user correct it later.
    taken_at = exif.extract_taken_at(original_bytes)
    if taken_at is None:
        taken_at = datetime.now()

    backend, path = storage_adapter.save_original(original_bytes, filename)

    with conn:  # one transaction per photo
        photo_id = repository.insert_photo(
            conn,
            picker_media_id=item["id"],
            taken_at=taken_at,
            # Google's own filename, not path's basename — the storage
            # adapter may have dedup-suffixed it, and Takeout keys on the
            # original (see migration 0002).
            original_filename=filename,
            original_storage_backend=backend,
            original_storage_path=path,
            # GPS intentionally absent — see module docstring.
        )

        if not detect_faces:
            return

        _detect_and_cluster_faces(
            conn,
            photo_id=photo_id,
            image_bytes=original_bytes,
            detect_faces=detect_faces,
            centroids=centroids,
            seed_faces=seed_faces,
            face_similarity_threshold=face_similarity_threshold,
            result=result,
        )


def _detect_and_cluster_faces(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    image_bytes: bytes,
    detect_faces: _FaceDetector,
    centroids: list[face_clustering.FaceClusterCentroid],
    seed_faces: list[tuple[int, str, list[float]]],
    face_similarity_threshold: float,
    result: IngestionResult,
) -> None:
    """Shared by both ingestion paths (Google Picker and local-device
    upload) — face detection/clustering/seed-matching doesn't depend on
    where the bytes came from, only on the image itself."""
    for bounding_box, embedding in detect_faces(image_bytes):
        result.faces_detected += 1

        cluster = face_clustering.assign(embedding, centroids, threshold=face_similarity_threshold)
        if cluster is None:
            suggested_name = face_clustering.find_seed_match(embedding, seed_faces)
            cluster_id = repository.insert_face_cluster(
                conn, representative_photo_id=photo_id, suggested_name=suggested_name
            )
            cluster = face_clustering.FaceClusterCentroid(cluster_id=cluster_id, centroid=list(embedding))
            centroids.append(cluster)
            result.face_clusters_created += 1
        else:
            cluster.add(embedding)

        repository.insert_face_instance(
            conn,
            photo_id=photo_id,
            face_cluster_id=cluster.cluster_id,
            bounding_box=bounding_box,
            embedding=embedding,
        )


# ---------------------------------------------------------------------
# Local-device import — the alternative to the Google Photos Picker flow
# ---------------------------------------------------------------------


def ingest_local_files(
    conn: sqlite3.Connection,
    files: list[tuple[str, bytes]],
    *,
    storage_adapter,
    detect_faces: Optional[_FaceDetector] = None,
    face_similarity_threshold: float = face_clustering.DEFAULT_SIMILARITY_THRESHOLD,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> IngestionResult:
    """Import photos the user picks from their own device's file input,
    rather than through Google's Picker API — no OAuth, no session
    polling, no dependency on a picker tab surviving in the background.

    Two real differences from ingest_picked_items():

      * Dedup key. The Picker API gives each item a stable picker_media_id;
        a local upload has no such id, so one is derived from the file's
        own content (a sha256 hash) — re-uploading the same file is a
        no-op rather than a duplicate, which is arguably a better property
        than filename-based dedup would have given anyway (a renamed copy
        of an already-imported photo is still recognized).

      * GPS. Local files were never round-tripped through Google's '=d'
        endpoint, so their GPS EXIF (if any) is still intact — this reads
        it directly and runs location clustering inline, skipping the
        separate Takeout-backfill step the Picker flow needs.
    """
    result = IngestionResult()

    centroids = [
        face_clustering.FaceClusterCentroid(cluster_id=cid, centroid=centroid, count=count)
        for cid, centroid, count in repository.load_face_cluster_centroids(conn)
    ]
    seed_faces = repository.load_seed_faces(conn) if detect_faces else []

    for filename, data in files:
        content_id = f"local-sha256:{hashlib.sha256(data).hexdigest()}"

        if repository.photo_id_for_picker_media_id(conn, content_id) is not None:
            result.skipped_already_imported += 1
            if on_progress:
                on_progress(filename, "already imported, skipped")
            continue

        try:
            _ingest_one_local(
                conn,
                filename=filename,
                data=data,
                content_id=content_id,
                storage_adapter=storage_adapter,
                detect_faces=detect_faces,
                centroids=centroids,
                seed_faces=seed_faces,
                face_similarity_threshold=face_similarity_threshold,
                result=result,
            )
        except Exception as e:  # noqa: BLE001 - per-photo guard, mirrors the Picker path
            conn.rollback()
            result.failed.append((filename, str(e)))
            if on_progress:
                on_progress(filename, f"FAILED: {e}")
            continue

        result.imported += 1
        if on_progress:
            on_progress(filename, "imported")

    if result.imported:
        cluster_locations(conn)

    return result


def _ingest_one_local(
    conn: sqlite3.Connection,
    *,
    filename: str,
    data: bytes,
    content_id: str,
    storage_adapter,
    detect_faces: Optional[_FaceDetector],
    centroids: list[face_clustering.FaceClusterCentroid],
    seed_faces: list[tuple[int, str, list[float]]],
    face_similarity_threshold: float,
    result: IngestionResult,
) -> None:
    taken_at = exif.extract_taken_at(data) or datetime.now()
    gps = exif.extract_gps(data)
    backend, path = storage_adapter.save_original(data, filename)

    with conn:  # one transaction per photo, same as the Picker path
        photo_id = repository.insert_photo(
            conn,
            picker_media_id=content_id,
            taken_at=taken_at,
            original_filename=filename,
            original_storage_backend=backend,
            original_storage_path=path,
            gps_lat=gps[0] if gps else None,
            gps_lng=gps[1] if gps else None,
        )

        if not detect_faces:
            return

        _detect_and_cluster_faces(
            conn,
            photo_id=photo_id,
            image_bytes=data,
            detect_faces=detect_faces,
            centroids=centroids,
            seed_faces=seed_faces,
            face_similarity_threshold=face_similarity_threshold,
            result=result,
        )


# ---------------------------------------------------------------------
# GPS backfill + location clustering (separate passes — see docstring)
# ---------------------------------------------------------------------


@dataclass
class BackfillResult:
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)


def backfill_gps_from_takeout(conn: sqlite3.Connection, takeout_dir: Path) -> BackfillResult:
    """Fills gps_lat/gps_lng from a Takeout export for photos that have no
    location yet. Re-runnable: photos that already have GPS (or a manual
    override) aren't revisited."""
    records = takeout.scan_takeout_export(takeout_dir)
    by_filename = {}
    for record in records:
        by_filename.setdefault(record.filename, record)

    result = BackfillResult()
    with conn:
        for photo_id, filename in repository.photos_missing_gps(conn):
            record = by_filename.get(filename)
            if record is None:
                result.unmatched.append(filename)
                continue
            repository.set_photo_gps(conn, photo_id, record.lat, record.lng)
            result.matched += 1
    return result


def cluster_locations(
    conn: sqlite3.Connection,
    *,
    threshold_m: float = location_clustering.DEFAULT_THRESHOLD_M,
) -> int:
    """Assigns every geotagged photo to a LocationCluster (§3 step 6),
    creating clusters as needed. Returns the number of clusters created.

    Runs as its own pass rather than inline during ingestion because GPS
    arrives later, via the Takeout backfill.
    """
    clusters = [
        location_clustering.LocationClusterCandidate(
            centroid_lat=lat, centroid_lng=lng, count=count
        )
        for _cid, lat, lng, count in repository.load_location_clusters(conn)
    ]
    cluster_ids = [cid for cid, _lat, _lng, _count in repository.load_location_clusters(conn)]
    created = 0

    with conn:
        for photo in repository.load_photos_for_anomaly_check(conn):
            before = len(clusters)
            cluster = location_clustering.assign_or_create(
                photo["lat"], photo["lng"], clusters, photo_index=photo["photo_id"], threshold_m=threshold_m
            )
            index = clusters.index(cluster)

            if len(clusters) > before:  # a new cluster was appended
                cluster_id = repository.insert_location_cluster(
                    conn, lat=cluster.centroid_lat, lng=cluster.centroid_lng
                )
                cluster_ids.append(cluster_id)
                created += 1
            else:
                cluster_id = cluster_ids[index]
                repository.update_location_cluster_centroid(
                    conn, cluster_id, cluster.centroid_lat, cluster.centroid_lng
                )

            repository.set_photo_location(conn, photo_id=photo["photo_id"], location_cluster_id=cluster_id)

    return created


# ---------------------------------------------------------------------
# Place-name suggestions (§3 step 6)
# ---------------------------------------------------------------------


def suggest_location_names(
    conn: sqlite3.Connection,
    *,
    use_geocoding: bool = True,
    use_ocr: bool = True,
    on_progress: Optional[Callable[[int, str, str], None]] = None,
) -> dict[str, int]:
    """Fills geocoded_name / ocr_name on location clusters that lack them.

    Both are *suggestions* shown next to an unlabeled cluster (§3 step 6);
    neither is ever written to LocationCluster.name, which stays exclusively
    what the user chose.

    Geocoding is batched once per cluster rather than per photo, which is
    what keeps it inside Nominatim's 1 req/sec policy. OCR only reads one
    photo per cluster — running the model over every photo of a place costs
    a lot for little extra signal, since a storefront usually appears in the
    first shot of it.
    """
    found = {"geocoded": 0, "ocr": 0}

    if use_geocoding:
        for cluster in repository.location_clusters_needing_suggestions(conn, source="geocode"):
            try:
                name = geocoding.reverse_geocode(cluster["centroid_lat"], cluster["centroid_lng"])
            except Exception:  # noqa: BLE001 - a hint is advisory, never fatal
                name = None
            with conn:
                repository.set_location_geocoded_name(conn, cluster["id"], name)
            if name:
                found["geocoded"] += 1
            if on_progress:
                on_progress(cluster["id"], "geocode", name or "(nothing)")

    if use_ocr:
        from . import ocr

        for cluster in repository.location_clusters_needing_suggestions(conn, source="ocr"):
            photos = repository.photos_in_location_cluster(conn, cluster["id"])
            candidates = []
            for photo in photos[:1]:
                path = Path(photo["path"])
                if not path.exists():
                    continue
                try:
                    candidates = ocr.read_text_candidates(path)
                except Exception:  # noqa: BLE001 - same, advisory only
                    candidates = []
            with conn:
                repository.set_location_ocr_name(conn, cluster["id"], ocr.encode_candidates(candidates))
            if candidates:
                found["ocr"] += 1
            if on_progress:
                on_progress(
                    cluster["id"],
                    "ocr",
                    ", ".join(c["text"] for c in candidates) or "(nothing)",
                )

    return found

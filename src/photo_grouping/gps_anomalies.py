"""Detects physically-impossible location jumps, for manual review.

Not in the original spec — added after real Takeout data turned up photos
timestamped 34 seconds apart with GPS ~300km apart (경기도 광주시 vs
전남 강진군), an implied ~31,700 km/h. That's a stale GPS lock in the
source data, not something our clustering can fix, but it silently
produces a wrong LocationCluster — so these photos should surface in a
manual review checklist rather than being trusted.

Whether the *same person appears in both* photos is the key confidence
signal. Two photos minutes apart from wildly different places are
perfectly normal on their own — a forwarded photo, a screenshot, a picture
someone sent you. But if the same face cluster appears in both, there is
positive evidence the same physical person was supposedly in two places at
once, so one of the two GPS readings must be wrong.

Rather than *filter* on that signal, anomalies are graded by it
(Confidence.HIGH when a face is shared, Confidence.LOW otherwise). Real
data showed why: filtering on shared faces missed a mislocated photo that
simply had nobody in it — 4 of 5 bad photos caught instead of 5 of 5.
Grading keeps full recall while still letting the checklist lead with the
cases that are certain.

Note this deliberately reports *pairs* and a per-photo conflict count, not
a verdict on which photo is wrong — a pair alone can't tell you that. The
conflict count is the useful signal for triage: in the real case above,
the mislocated photos each conflicted with many correctly-located
neighbours (17 vs 4), so ranking a checklist by conflict count puts the
true outliers on top.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from .location_clustering import haversine_distance_m


class Confidence(enum.Enum):
    """HIGH: the same face cluster appears in both photos, so the same
    person was supposedly in two places at once — one reading must be
    wrong. LOW: the jump is impossible, but nothing ties the two photos to
    the same subject, so they may be unrelated (a forwarded image, a
    screenshot) rather than a GPS error."""

    HIGH = "high"
    LOW = "low"

# Roughly high-speed rail (KTX tops out ~300 km/h). Anything faster than
# this between two photos of the same person is implausible over the short
# windows this check looks at. Air travel can legitimately exceed it, so
# raise this if a frequent flyer sees false positives — the failure mode is
# a slightly longer manual checklist, not wrong data.
DEFAULT_MAX_PLAUSIBLE_SPEED_KMH = 300.0

# Only look at photos close together in time; over hours, almost any
# distance becomes plausible and the check stops being informative.
DEFAULT_MAX_TIME_DELTA_SECONDS = 600.0  # 10 minutes

# Ordinary GPS jitter (and photos with identical timestamps) shouldn't ever
# trip this. Below this distance, no pair is ever flagged regardless of
# implied speed.
DEFAULT_GPS_NOISE_TOLERANCE_M = 100.0


@dataclass
class PhotoLocationSample:
    """The subset of a Photo row this check needs. Kept as its own type so
    the detection logic doesn't depend on the DB layer — callers build
    these from Photo/FaceInstance/PhotoLocation rows (or, before the
    pipeline is DB-wired, straight from Takeout records + face clusters)."""

    photo_id: object  # int row id in real use; any hashable identifier in tests/spikes
    taken_at: Optional[datetime]
    lat: Optional[float]
    lng: Optional[float]
    face_cluster_ids: frozenset = field(default_factory=frozenset)


@dataclass
class GpsAnomaly:
    photo_a_id: object
    photo_b_id: object
    distance_m: float
    time_delta_seconds: float
    implied_speed_kmh: float
    shared_face_cluster_ids: frozenset
    confidence: Confidence


def _implied_speed_kmh(distance_m: float, time_delta_seconds: float) -> float:
    if time_delta_seconds <= 0:
        return float("inf")  # same instant, different place
    return (distance_m / 1000.0) / (time_delta_seconds / 3600.0)


def find_gps_anomalies(
    photos: Iterable[PhotoLocationSample],
    *,
    max_time_delta_seconds: float = DEFAULT_MAX_TIME_DELTA_SECONDS,
    max_plausible_speed_kmh: float = DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
    gps_noise_tolerance_m: float = DEFAULT_GPS_NOISE_TOLERANCE_M,
    require_shared_face: bool = False,
) -> list[GpsAnomaly]:
    """Returns pairs of photos whose locations are too far apart to be
    reachable in the time between them, each graded by Confidence.

    By default every impossible jump is reported, graded HIGH when the two
    photos share a face cluster and LOW when they don't — filtering on
    shared faces instead loses real errors in photos that simply have
    nobody in them (see module docstring). Pass require_shared_face=True to
    get only the HIGH-confidence pairs.
    """
    ordered = sorted(
        (p for p in photos if p.taken_at is not None and p.lat is not None and p.lng is not None),
        key=lambda p: p.taken_at,
    )

    anomalies: list[GpsAnomaly] = []
    for i, photo_a in enumerate(ordered):
        for photo_b in ordered[i + 1 :]:
            time_delta = (photo_b.taken_at - photo_a.taken_at).total_seconds()
            # Sorted by time, so once we're past the window every later
            # photo is too — stop scanning this photo's neighbours.
            if time_delta > max_time_delta_seconds:
                break

            shared = photo_a.face_cluster_ids & photo_b.face_cluster_ids
            if require_shared_face and not shared:
                continue

            distance = haversine_distance_m(photo_a.lat, photo_a.lng, photo_b.lat, photo_b.lng)
            if distance <= gps_noise_tolerance_m:
                continue

            speed = _implied_speed_kmh(distance, time_delta)
            if speed <= max_plausible_speed_kmh:
                continue

            anomalies.append(
                GpsAnomaly(
                    photo_a_id=photo_a.photo_id,
                    photo_b_id=photo_b.photo_id,
                    distance_m=distance,
                    time_delta_seconds=time_delta,
                    implied_speed_kmh=speed,
                    shared_face_cluster_ids=frozenset(shared),
                    confidence=Confidence.HIGH if shared else Confidence.LOW,
                )
            )
    return anomalies


@dataclass
class ReviewItem:
    photo_id: object
    conflict_count: int
    confidence: Confidence  # highest confidence among this photo's conflicts


def photos_needing_review(anomalies: Iterable[GpsAnomaly]) -> list[ReviewItem]:
    """Reduces anomaly pairs to a review checklist, most-conflicted first,
    with HIGH confidence breaking ties ahead of LOW.

    Conflict count leads because it is the stronger predictor of which
    photo is actually wrong: a photo whose location disagrees with many of
    its temporal neighbours is geographically isolated from its whole
    cohort, while a photo caught in a single pair may just be adjacent to
    someone else's error. On real data the true outliers had 23 conflicts
    each against 5 for the collateral — a clean split that ranking by
    confidence first would have destroyed, burying a genuine 23-conflict
    outlier (a photo with no faces in it, hence LOW) below 17 correctly
    located photos.

    Confidence is the tiebreaker rather than the primary key for that
    reason: it grades how much a *single pair* can be trusted, whereas
    conflict count aggregates evidence across many pairs. A photo inherits
    the highest confidence among its conflicts — one HIGH pair is enough to
    make it a certain error, however many LOW pairs surround it."""
    counts: dict[object, int] = {}
    best_confidence: dict[object, Confidence] = {}
    for anomaly in anomalies:
        for photo_id in (anomaly.photo_a_id, anomaly.photo_b_id):
            counts[photo_id] = counts.get(photo_id, 0) + 1
            if anomaly.confidence is Confidence.HIGH:
                best_confidence[photo_id] = Confidence.HIGH
            else:
                best_confidence.setdefault(photo_id, Confidence.LOW)

    items = [
        ReviewItem(photo_id=photo_id, conflict_count=count, confidence=best_confidence[photo_id])
        for photo_id, count in counts.items()
    ]
    return sorted(
        items,
        key=lambda item: (-item.conflict_count, 0 if item.confidence is Confidence.HIGH else 1, str(item.photo_id)),
    )

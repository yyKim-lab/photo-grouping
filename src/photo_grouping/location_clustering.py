"""Location clustering — incremental, centroid-based (§3 step 6, §6 step 6).

Unlike face clustering (batch DBSCAN over embeddings, §6 step 5), the spec
describes location clustering as incremental: each new photo's GPS is
compared against *existing* LocationCluster centroids and either joins the
nearest one (if within threshold) or starts a new cluster. This matches how
ingestion actually runs — one photo at a time, against a growing set of
clusters already in the database — so there's no batch-DBSCAN step to
tune here the way there was for faces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EARTH_RADIUS_M = 6_371_000  # mean radius, meters — good enough for a ~150m threshold

DEFAULT_THRESHOLD_M = 150  # spec §3 step 6's suggested starting value


def haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class LocationClusterCandidate:
    """In-memory stand-in for a LocationCluster row plus enough state to
    keep its centroid accurate as more photos join it — a running mean of
    every point assigned so far, not just the first point. The real
    ingestion pipeline (once DB-wired) persists centroid_lat/centroid_lng
    on the LocationCluster row itself instead of tracking `count` in Python."""

    centroid_lat: float
    centroid_lng: float
    count: int = 1
    photo_indices: list[int] = field(default_factory=list)

    def add(self, lat: float, lng: float, photo_index: int) -> None:
        # Running mean, weighted by how many points already contributed —
        # keeps the centroid representative without storing every point.
        self.centroid_lat = (self.centroid_lat * self.count + lat) / (self.count + 1)
        self.centroid_lng = (self.centroid_lng * self.count + lng) / (self.count + 1)
        self.count += 1
        self.photo_indices.append(photo_index)


def assign_or_create(
    lat: float,
    lng: float,
    clusters: list[LocationClusterCandidate],
    photo_index: int,
    threshold_m: float = DEFAULT_THRESHOLD_M,
) -> LocationClusterCandidate:
    """Nearest-centroid assignment: joins the closest cluster within
    threshold_m, or appends a new singleton cluster to `clusters` and
    returns it. Mutates `clusters` in place, matching how the real
    ingestion pipeline processes one photo at a time against the clusters
    already in the database."""
    nearest = None
    nearest_distance = float("inf")
    for cluster in clusters:
        distance = haversine_distance_m(lat, lng, cluster.centroid_lat, cluster.centroid_lng)
        if distance < nearest_distance:
            nearest, nearest_distance = cluster, distance

    if nearest is not None and nearest_distance <= threshold_m:
        nearest.add(lat, lng, photo_index)
        return nearest

    new_cluster = LocationClusterCandidate(centroid_lat=lat, centroid_lng=lng, photo_indices=[photo_index])
    clusters.append(new_cluster)
    return new_cluster

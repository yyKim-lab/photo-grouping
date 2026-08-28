"""Incremental face-cluster assignment (§3 steps 5-6).

Mirrors location_clustering.py's shape, and for the same reason: ingestion
processes one photo at a time against clusters already in the database, so
it needs nearest-centroid assignment, not a batch algorithm.

Metric is cosine similarity, matching ArcFace's L2-normalized embeddings
(see face_embeddings.py). Higher is more alike, so the comparison is
`>= threshold`, the opposite direction from a distance metric — worth
noting because the earlier dlib implementation used Euclidean distance and
compared `<=`.

Cluster centroids are a running mean of member embeddings. That mean is not
itself unit-length, so it is re-normalized at comparison time; skipping
that would let a cluster's similarity scores drift downward purely as a
function of how varied its members are.

One known limitation, measured on real data: because a face is compared
against a centroid *at that moment*, two clusters can each drift toward the
other after they are created and end up closer than the threshold, with
nothing re-checking them. A periodic consolidation pass is the fix; it does
not exist yet. Do not add one that merges purely on centroid proximity
without checking real photos first — under the previous dlib model the
closest cluster pairs in this library were confirmed *different people*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Tuned against user-assigned labels on this library — 529 same-person pairs
# and 2711 different-people pairs — rather than against InsightFace's generic
# guidance (~0.35) or the same-photo-violation heuristic.
#
# Both of those understated the threshold badly here. Same-photo violations
# only catch merges between people who appear *together* in a frame, so they
# are blind to exactly the failure this library suffers from: close family
# resemblance. At 0.35 the measured false-merge rate was 5.5%, including
# mother against child (+0.428) and sister against sister (+0.361).
#
# Measured distributions: within-person similarity had median 0.714, while
# the single worst different-people pair reached 0.477. 0.50 therefore clears
# every observed stranger pair (0.0% false merges) while still linking 83% of
# same-person pairs directly — and cluster-level recall runs higher than that
# pairwise figure, since a person's faces only need to chain together, not to
# all pairwise exceed the threshold.
#
# Raising this splits more; lowering it starts merging relatives. Re-measure
# against real labels if the model, detection resolution, or size floor
# changes — and prefer labels over the same-photo heuristic when both exist.
DEFAULT_SIMILARITY_THRESHOLD = 0.50


@dataclass
class FaceClusterCentroid:
    """In-memory running centroid for one FaceCluster row.

    The schema has no centroid column on face_cluster (unlike
    location_cluster, which stores centroid_lat/lng), so centroids are
    rebuilt from member embeddings when an ingestion run starts and kept
    current in memory as faces are assigned. At personal scale that costs
    one query per run and avoids a migration; if libraries ever get large
    enough for that load to matter, a stored centroid column is the fix.
    """

    cluster_id: object
    centroid: Sequence[float]
    count: int = 1
    member_face_ids: list = field(default_factory=list)

    def add(self, embedding: Sequence[float], face_id: object = None) -> None:
        """Folds one more embedding into the running mean."""
        self.centroid = [
            (existing * self.count + new) / (self.count + 1)
            for existing, new in zip(self.centroid, embedding)
        ]
        self.count += 1
        if face_id is not None:
            self.member_face_ids.append(face_id)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 1 is identical direction.

    Normalizes both sides rather than assuming unit length: individual
    embeddings arrive normalized, but a cluster centroid is their mean and
    is not."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norms = np.linalg.norm(va) * np.linalg.norm(vb)
    if norms == 0:
        return 0.0
    return float(np.dot(va, vb) / norms)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Straight L2 distance. Not used for assignment any more — kept because
    it is the natural metric for un-normalized embeddings, should another
    model ever be plugged in."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def find_nearest(
    embedding: Sequence[float],
    clusters: Sequence[FaceClusterCentroid],
) -> tuple[Optional[FaceClusterCentroid], float]:
    """Returns (most similar cluster, similarity), or (None, -1.0) if there
    are no clusters yet."""
    nearest: Optional[FaceClusterCentroid] = None
    best_similarity = -1.0
    for cluster in clusters:
        similarity = cosine_similarity(embedding, cluster.centroid)
        if similarity > best_similarity:
            nearest, best_similarity = cluster, similarity
    return nearest, best_similarity


def assign(
    embedding: Sequence[float],
    clusters: Sequence[FaceClusterCentroid],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Optional[FaceClusterCentroid]:
    """Returns the cluster this face belongs to, or None if it is unlike
    every existing cluster and the caller should create a new one. Does not
    mutate `clusters` — creating a cluster needs a database row first, so
    the caller owns that step and then calls `.add()`."""
    nearest, similarity = find_nearest(embedding, clusters)
    if nearest is not None and similarity >= threshold:
        return nearest
    return None


# Seed matching (§3 step 5, §4.5) is deliberately stricter than ordinary
# cluster assignment: the spec calls out seed embeddings — cropped from a
# small screenshot thumbnail, not a full photo — as less reliable, and a
# wrong seed match would mislabel a stranger with a real person's name
# rather than just fragmenting one person across two clusters. Padding
# above DEFAULT_SIMILARITY_THRESHOLD reflects that asymmetry: a missed
# match costs one extra queue confirmation, a false one costs a wrong name.
DEFAULT_SEED_MATCH_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD + 0.05


def find_seed_match(
    embedding: Sequence[float],
    seed_faces: Sequence[tuple[int, str, Sequence[float]]],
    *,
    threshold: float = DEFAULT_SEED_MATCH_THRESHOLD,
) -> Optional[str]:
    """seed_faces is (id, name, embedding) tuples, as returned by
    repository.load_seed_faces. Returns the best-matching name above
    threshold, or None — never auto-assigned, only ever offered as a
    suggestion (see migration 0008)."""
    best_name: Optional[str] = None
    best_similarity = threshold
    for _id, name, seed_embedding in seed_faces:
        similarity = cosine_similarity(embedding, seed_embedding)
        if similarity >= best_similarity:
            best_name, best_similarity = name, similarity
    return best_name

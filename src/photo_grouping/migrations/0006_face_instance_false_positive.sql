-- Adds false_positive_at to face_instance: a detection the user marked as
-- not actually a face (§4.4's "remove (mark as false-positive detection)").
--
-- A nullable timestamp, not a delete, matching the app's existing pattern
-- for hidden clusters (excluded_at, migration 0004): non-destructive,
-- auditable, and cheap to reverse if the user marks the wrong crop by
-- mistake. Physical deletion was the other option, but the spec's own
-- wording ("mark as") points at a flag, and it costs nothing here since
-- face_instance rows are cheap and never referenced by anything that would
-- make a stale row dangerous (unlike, say, a photo file on disk).
--
-- Every query that aggregates or lists faces for clustering, counting, or
-- display must exclude false_positive_at IS NOT NULL rows — a false
-- positive isn't a face, so it must never contribute to a centroid, an
-- instance count, or the labeling queue's regrow threshold.

ALTER TABLE face_instance ADD COLUMN false_positive_at TEXT;

CREATE INDEX idx_face_instance_false_positive ON face_instance (false_positive_at);

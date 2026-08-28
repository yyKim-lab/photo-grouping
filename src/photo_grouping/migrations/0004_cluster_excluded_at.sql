-- Adds excluded_at to face_cluster and location_cluster: clusters the user
-- has said they don't want to see — strangers caught in the background of a
-- photo, passers-by, a face on a poster.
--
-- Deliberately a nullable timestamp rather than a fourth `status` value,
-- for two reasons:
--
--   * Exclusion is orthogonal to labeling state, not another point on the
--     same axis. A cluster can be unlabeled *and* excluded (a stranger), or
--     named *and* excluded (someone you can identify but don't want in your
--     albums). Folding it into `status` would make those unrepresentable.
--
--   * Changing a CHECK constraint in SQLite requires rebuilding the table
--     and re-pointing foreign keys, which is a lot of risk for a flag.
--
-- Storing the timestamp rather than a boolean costs nothing and records
-- when the decision was made, which matters if a user later wants to review
-- what they hid.

ALTER TABLE face_cluster ADD COLUMN excluded_at TEXT;
ALTER TABLE location_cluster ADD COLUMN excluded_at TEXT;

CREATE INDEX idx_face_cluster_excluded ON face_cluster (excluded_at);
CREATE INDEX idx_location_cluster_excluded ON location_cluster (excluded_at);

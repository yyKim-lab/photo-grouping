-- Adds last_shown_at / instance_count_at_last_shown to location_cluster.
--
-- §2's pseudocode lists these two columns on FaceCluster but not on
-- LocationCluster. §4.2 nonetheless applies the same rule to both: the
-- labeling queue holds unlabeled clusters plus not_sure clusters that have
-- grown by >= 5 instances since they were last shown, and its card prompt
-- is "Do you know this person?" / "Do you know this place?" — places go
-- through the identical flow.
--
-- Without these columns a place marked "not sure" could never re-surface,
-- no matter how many more photos landed in it, so the omission reads as an
-- oversight in the pseudocode rather than a deliberate asymmetry.

ALTER TABLE location_cluster ADD COLUMN last_shown_at TEXT;
ALTER TABLE location_cluster ADD COLUMN instance_count_at_last_shown INTEGER NOT NULL DEFAULT 0;

-- Adds suggested_name to face_cluster, for §4.5's seed-face matching.
--
-- §3 step 5 says a seed match during ingestion must be "surfaced as a
-- high-confidence suggestion pending user confirm" rather than
-- auto-assigned — seed embeddings come from low-res screenshot crops and
-- are explicitly called out as less reliable than full-photo embeddings.
-- This column is where that suggestion lives until the queue's normal
-- naming flow confirms or rejects it: the labeling queue shows it as a
-- tappable chip (the same suggestion_chips UI already used for
-- geocoded/OCR place names), never written to `name` automatically.

ALTER TABLE face_cluster ADD COLUMN suggested_name TEXT;

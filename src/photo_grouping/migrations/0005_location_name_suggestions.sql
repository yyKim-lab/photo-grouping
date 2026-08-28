-- Adds name suggestions to location_cluster, from two independent sources.
--
-- §3 step 6 says a cluster gets a reverse-geocoded name "shown alongside
-- the unlabeled cluster ... not the same as user-assigned name". The
-- geocoding module existed but nothing ever stored or displayed its output,
-- so the UI was showing raw coordinates.
--
-- Two columns rather than one, because the sources are genuinely different
-- kinds of evidence and neither should clobber the other:
--
--   geocoded_name  what the map provider calls this coordinate. Reliable
--                  for administrative areas, weaker for venues.
--   ocr_name       text read off the photo itself — a storefront sign, a
--                  banner. Often the *actual* name of a place when the map
--                  only knows the street ("코롬방제과점" rather than
--                  "영산로75번길, 목포시"), but also often noise.
--
-- Both are suggestions offered to the user, never applied automatically:
-- LocationCluster.name stays exclusively what the user chose.
--
-- geocoded_at / ocr_at record that a pass ran, so a cluster where the
-- source genuinely had nothing to say isn't retried on every run.

ALTER TABLE location_cluster ADD COLUMN geocoded_name TEXT;
ALTER TABLE location_cluster ADD COLUMN geocoded_at TEXT;
ALTER TABLE location_cluster ADD COLUMN ocr_name TEXT;
ALTER TABLE location_cluster ADD COLUMN ocr_at TEXT;

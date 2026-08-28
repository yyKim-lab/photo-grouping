-- Adds photo.original_filename: the filename as Google Photos reports it
-- (PickedMediaItem.mediaFile.filename).
--
-- Why this can't be derived from original_storage_path, which is what the
-- code did before: the storage adapters deliberately refuse to overwrite an
-- existing file, appending "_1", "_2" and so on when two photos arrive with
-- the same name. That is correct behavior — Google hands back camera-default
-- names like IMG_0001.jpg for genuinely different photos from different
-- devices — but it means the stored path's basename is not reliably the
-- original filename.
--
-- That matters because Google Takeout sidecars key on the *original*
-- filename, so GPS backfill (see takeout.py) silently failed to match any
-- photo whose file had been dedup-suffixed on the way to disk.
--
-- Nullable rather than NOT NULL: rows written before this migration have no
-- record of their original filename beyond the path, so they are backfilled
-- best-effort below (correct for every photo that never hit a collision,
-- which is the common case). Code written after this migration always
-- populates it explicitly.

ALTER TABLE photo ADD COLUMN original_filename TEXT;

-- Best-effort backfill for pre-existing rows: take the path's basename.
-- Photos that were dedup-suffixed keep the suffixed name here and will
-- still fail to match Takeout — re-ingesting them is the fix, and is
-- cheap since ingestion skips on picker_media_id.
UPDATE photo
SET original_filename = replace(original_storage_path, rtrim(original_storage_path, replace(original_storage_path, '/', '')), '')
WHERE original_filename IS NULL;

CREATE INDEX idx_photo_original_filename ON photo (original_filename);

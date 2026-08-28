-- Two more app-wide settings, same table as migration 0011:
--   narrative_language — what language Autobio/Diary entries are written
--     in. Previously the prompt just said "write in whichever language
--     the given names/places/notes/events are mostly in", which produced
--     entries that flipped between Korean and English day to day
--     depending on what happened to be in the photo metadata. This makes
--     it an explicit, stable choice instead.
--   ui_language — what language the app's own interface (nav, buttons,
--     labels) is shown in. Independent from narrative_language: someone
--     may want to read/write their diary in Korean while running the app
--     itself in English, or vice versa.
-- Both default to Korean, matching this app's primary use so far.
ALTER TABLE app_settings ADD COLUMN narrative_language TEXT NOT NULL DEFAULT 'ko';
ALTER TABLE app_settings ADD COLUMN ui_language TEXT NOT NULL DEFAULT 'ko';

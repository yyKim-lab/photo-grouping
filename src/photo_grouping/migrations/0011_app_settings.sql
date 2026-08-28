-- General app-wide settings (distinct from autobio_settings, migration
-- 0010, which is Autobio-specific) — starts with just the speech-
-- recognition language for §4.2's voice-typing labeling input. Single-row
-- table, same pattern as pending_import_session / autobio_settings.
CREATE TABLE app_settings (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    speech_language  TEXT NOT NULL DEFAULT 'ko-KR'
);

INSERT INTO app_settings (id, speech_language) VALUES (1, 'ko-KR');

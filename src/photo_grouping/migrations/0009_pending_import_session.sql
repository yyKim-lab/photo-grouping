-- Persists the in-progress Google Photos Picker session server-side,
-- independent of any one browser tab.
--
-- Before this, the only place a picker session's id/pickerUri lived was in
-- that one tab's DOM (a hidden form field) and JS closure — if the picker
-- opened in the same tab instead of a new one (observed in practice: "no
-- tab to switch back to" after finishing in Google's picker UI), that page
-- was gone and the session was unrecoverable from the app's UI, even
-- though Google still had the picked items waiting to be fetched. A single
-- row here means /import can always offer "finish this" regardless of
-- which tab, window, or browser the user comes back in.
--
-- Single-user, sequential-imports app: only one pending session is ever
-- meaningful at a time, so this is a one-row table (id fixed at 1),
-- replaced wholesale by each new /import/start rather than accumulating
-- history.
CREATE TABLE pending_import_session (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    session_id   TEXT NOT NULL,
    picker_uri   TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

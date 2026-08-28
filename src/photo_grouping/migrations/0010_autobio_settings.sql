-- §4.6: user-toggleable setting for whether the "N unlabeled people/
-- places appear today" nudge shows on an Autobio entry page. Decided
-- default per spec: ON. Single-row table, same pattern as
-- pending_import_session (migration 0009) — one flag, one row.
CREATE TABLE autobio_settings (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    show_unlabeled_nudge  INTEGER NOT NULL DEFAULT 1 CHECK (show_unlabeled_nudge IN (0, 1))
);

INSERT INTO autobio_settings (id, show_unlabeled_nudge) VALUES (1, 1);

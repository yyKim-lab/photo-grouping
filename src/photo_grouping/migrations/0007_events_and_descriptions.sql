-- Adds Event labeling (new — not in the original spec) and free-text
-- descriptions on clusters and photos (also new).
--
-- Events are purely user-created, unlike FaceCluster/LocationCluster: there
-- is no detector proposing them, the user selects a batch of photos and
-- names the occasion directly ("Gapyeong trip", "코롬방 방문"). That is why
-- events get their own table rather than being folded into location
-- clusters or some existing enum — an event is not a place or a person,
-- it is an arbitrary grouping the user defines.
--
-- photo_event is many-to-many (a photo can belong to more than one event —
-- e.g. an all-day outing that was also "someone's birthday"), unlike
-- photo_location which is one-per-photo by nature (a photo has exactly one
-- place it was taken). A photo can appear in zero, one, or several events.
--
-- Descriptions are free text on face_cluster, location_cluster, and photo.
-- Nullable, no length constraint — this is prose, not a label.

CREATE TABLE event (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE photo_event (
    photo_id   INTEGER NOT NULL REFERENCES photo (id) ON DELETE CASCADE,
    event_id   INTEGER NOT NULL REFERENCES event (id) ON DELETE CASCADE,
    added_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (photo_id, event_id)
);

CREATE INDEX idx_photo_event_event_id ON photo_event (event_id);

ALTER TABLE face_cluster ADD COLUMN description TEXT;
ALTER TABLE location_cluster ADD COLUMN description TEXT;
ALTER TABLE photo ADD COLUMN description TEXT;

-- Photo Grouping App — initial schema
-- Implements the data model from the implementation spec, §2.
--
-- Design notes (decisions made beyond the spec's literal field lists, since
-- SQLite needs concrete types/keys the spec's pseudocode doesn't spell out):
--
-- * Every table gets an INTEGER PRIMARY KEY `id`, even where §2's pseudocode
--   didn't list one (FaceInstance, PhotoLocation). FaceInstance rows are
--   individually reassigned during split/merge (§4.3), so they need their
--   own addressable id. PhotoLocation is one-per-photo (a photo has at most
--   one location cluster), so `photo_id` is both the primary key and the FK.
-- * SQLite has no native ENUM — every ENUM field from §2 becomes
--   TEXT + CHECK (col IN (...)).
-- * `embedding` columns are BLOB: a packed little-endian float32 array
--   (Python: `array.array('f', values).tobytes()`), decoded the same way.
--   Chosen over JSON text for compactness — these run through DBSCAN over
--   the full library, and stdlib `array` avoids a numpy dependency.
-- * `bounding_box` is TEXT (JSON: {"x","y","width","height"}, normalized
--   0..1 fractions of image width/height), consistent with the other JSON
--   fields the spec already stores as TEXT.
-- * ClusterEvent needs a `cluster_type` column ('face' | 'location') that
--   §2's pseudocode omits. source_cluster_ids/resulting_cluster_ids can
--   reference either FaceCluster or LocationCluster rows (§4.3 merge/split
--   applies to both), so without this column there's no way to know which
--   table the ids in a given event point into.
-- * created_at/updated_at timestamps are added throughout as standard
--   bookkeeping; the spec doesn't mention them but doesn't preclude them
--   either, and they're needed for `last_shown_at`-style logic elsewhere
--   to make sense of row history.
--
-- Explicitly NOT included here (out of scope for "just the DB schema"):
-- no multi-tenancy / users table. Each deployment of this app is a single
-- local install for one person (per the onboarding flow in §5) — "other
-- users" means separate installs, not shared rows in one database.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- Photo
-- ---------------------------------------------------------------------
CREATE TABLE photo (
    id                          INTEGER PRIMARY KEY,
    picker_media_id             TEXT NOT NULL UNIQUE,   -- Google PickedMediaItem.id
    taken_at                    TEXT NOT NULL,           -- ISO 8601, from EXIF
    taken_at_override           TEXT,                    -- ISO 8601, §4.4 manual correction

    gps_lat                     REAL,
    gps_lng                     REAL,
    gps_override_lat            REAL,                    -- §4.4 manual correction, supersedes EXIF
    gps_override_lng            REAL,

    original_storage_backend    TEXT NOT NULL
                                 CHECK (original_storage_backend IN
                                        ('local', 'icloud', 'google_drive', 'dropbox', 's3')),
    original_storage_path       TEXT NOT NULL,           -- filesystem path or cloud file id/path

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_photo_taken_at ON photo (taken_at);

-- ---------------------------------------------------------------------
-- FaceCluster
-- ---------------------------------------------------------------------
CREATE TABLE face_cluster (
    id                          INTEGER PRIMARY KEY,
    status                      TEXT NOT NULL DEFAULT 'unlabeled'
                                 CHECK (status IN ('unlabeled', 'not_sure', 'named')),
    name                        TEXT,
    representative_photo_id     INTEGER
                                 REFERENCES photo (id) ON DELETE SET NULL,
    last_shown_at               TEXT,
    instance_count_at_last_shown INTEGER NOT NULL DEFAULT 0,

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_face_cluster_status ON face_cluster (status);

-- ---------------------------------------------------------------------
-- FaceInstance
-- ---------------------------------------------------------------------
CREATE TABLE face_instance (
    id                          INTEGER PRIMARY KEY,
    photo_id                    INTEGER NOT NULL
                                 REFERENCES photo (id) ON DELETE CASCADE,
    face_cluster_id             INTEGER NOT NULL
                                 REFERENCES face_cluster (id) ON DELETE CASCADE,
    bounding_box                TEXT NOT NULL,           -- JSON {x, y, width, height}, 0..1 fractions
    embedding                   BLOB NOT NULL,            -- packed float32 array, see header note
    is_manual_override          INTEGER NOT NULL DEFAULT 0
                                 CHECK (is_manual_override IN (0, 1)),

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_face_instance_photo_id ON face_instance (photo_id);
CREATE INDEX idx_face_instance_face_cluster_id ON face_instance (face_cluster_id);

-- ---------------------------------------------------------------------
-- LocationCluster
-- ---------------------------------------------------------------------
CREATE TABLE location_cluster (
    id                          INTEGER PRIMARY KEY,
    status                      TEXT NOT NULL DEFAULT 'unlabeled'
                                 CHECK (status IN ('unlabeled', 'not_sure', 'named')),
    name                        TEXT,
    centroid_lat                REAL NOT NULL,
    centroid_lng                REAL NOT NULL,

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_location_cluster_status ON location_cluster (status);

-- ---------------------------------------------------------------------
-- PhotoLocation (one row per photo that has a location assignment;
-- photo_id is both PK and FK — see header note)
-- ---------------------------------------------------------------------
CREATE TABLE photo_location (
    photo_id                    INTEGER PRIMARY KEY
                                 REFERENCES photo (id) ON DELETE CASCADE,
    location_cluster_id         INTEGER NOT NULL
                                 REFERENCES location_cluster (id) ON DELETE CASCADE,
    is_manual_override          INTEGER NOT NULL DEFAULT 0
                                 CHECK (is_manual_override IN (0, 1)),

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_photo_location_cluster_id ON photo_location (location_cluster_id);

-- ---------------------------------------------------------------------
-- ClusterEvent
-- ---------------------------------------------------------------------
CREATE TABLE cluster_event (
    id                          INTEGER PRIMARY KEY,
    cluster_type                TEXT NOT NULL
                                 CHECK (cluster_type IN ('face', 'location')),
    event_type                  TEXT NOT NULL
                                 CHECK (event_type IN ('merge', 'split')),
    source_cluster_ids          TEXT NOT NULL,           -- JSON array of ids
    resulting_cluster_ids       TEXT NOT NULL,           -- JSON array of ids
    timestamp                   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_cluster_event_type ON cluster_event (cluster_type, event_type);

-- ---------------------------------------------------------------------
-- SeedFace
-- ---------------------------------------------------------------------
CREATE TABLE seed_face (
    id                          INTEGER PRIMARY KEY,
    name                        TEXT NOT NULL,
    embedding                   BLOB NOT NULL,           -- packed float32 array, see header note
    source                      TEXT NOT NULL
                                 CHECK (source IN ('screenshot_import')),
    face_cluster_id             INTEGER
                                 REFERENCES face_cluster (id) ON DELETE SET NULL,

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_seed_face_face_cluster_id ON seed_face (face_cluster_id);

-- ---------------------------------------------------------------------
-- AutobioEntry
-- ---------------------------------------------------------------------
CREATE TABLE autobio_entry (
    id                          INTEGER PRIMARY KEY,
    date                        TEXT NOT NULL UNIQUE,   -- ISO 8601 date, one entry per day
    draft_text                  TEXT NOT NULL DEFAULT '',
    final_text                  TEXT NOT NULL DEFAULT '',
    has_unlabeled_flag          INTEGER NOT NULL DEFAULT 0
                                 CHECK (has_unlabeled_flag IN (0, 1)),
    segments                    TEXT NOT NULL DEFAULT '[]',
                                 -- JSON array of
                                 -- {id, text, source_photo_ids: [], edited: bool}

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------
-- AutobioSummary
-- ---------------------------------------------------------------------
CREATE TABLE autobio_summary (
    id                          INTEGER PRIMARY KEY,
    start_date                  TEXT NOT NULL,           -- ISO 8601 date
    end_date                    TEXT NOT NULL,            -- ISO 8601 date
    source_entry_ids            TEXT NOT NULL,            -- JSON array of autobio_entry.id
    narrative_text               TEXT NOT NULL DEFAULT '',

    created_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at                  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_autobio_summary_range ON autobio_summary (start_date, end_date);

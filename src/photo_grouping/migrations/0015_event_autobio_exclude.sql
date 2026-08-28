-- Lets a user exclude a whole event/group from Autobio and Diary
-- generation — for something they don't want an AI ever describing (a
-- private appointment, a work event, etc.). A photo tagged with an
-- excluded event is dropped from repository.photos_for_date() entirely,
-- not just its event label hidden from the prompt — see that function's
-- docstring update. Independent of face_cluster/location_cluster's
-- existing excluded_at (§4.3, "not a person/place worth labeling") —
-- this is about generated *text*, not about the labeling queue.
ALTER TABLE event ADD COLUMN excluded_from_autobio INTEGER NOT NULL DEFAULT 0
    CHECK (excluded_from_autobio IN (0, 1));

-- Where imported photo originals are saved on disk (§5 LocalStorageAdapter's
-- root_dir), user-configurable instead of hardcoded to web.py's
-- DEFAULT_ORIGINALS_DIR. '' means "use the default" — same empty-string-
-- means-default convention as the other app_settings columns (ui_language,
-- llm_provider, etc).
--
-- Only affects where FUTURE imports land: changing this does not move
-- already-imported files or update their existing original_storage_path
-- rows — see web.py's settings_originals_dir_save() docstring.
ALTER TABLE app_settings ADD COLUMN originals_dir TEXT NOT NULL DEFAULT '';

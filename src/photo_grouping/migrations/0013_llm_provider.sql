-- Persistent "which AI provider" choice for self-hosters who've configured
-- both an Anthropic and an OpenAI key and want to pick one from Settings
-- rather than setting the AUTOBIO_LLM_PROVIDER env var (llm.py still
-- honors that env var too, and it still wins if set — this is the
-- friendlier, no-terminal-needed equivalent for everyone else).
-- '' means "auto-detect" (llm.py's original behavior: Anthropic if
-- configured, else OpenAI) — same empty-string-means-default convention
-- as the other app_settings columns.
ALTER TABLE app_settings ADD COLUMN llm_provider TEXT NOT NULL DEFAULT '';

# Photo Grouping App

A self-hosted photo organizer: ingest photos from Google Photos or straight
from your device, automatically group them by the people and places in
them, and optionally have an AI draft a daily diary-style narrative from
what it sees.

**Platform:** a local web app you run on your own machine — Python
backend, browser UI. Not a hosted service; every install has its own
SQLite database and its own credentials. Nothing about your photos or
your AI/Google API keys is shared with anyone else's install.

## Features

- **Import** from Google Photos (via the Picker API) or directly from
  files already on your device.
- **Face clustering** — groups photos by who's in them (InsightFace/
  ArcFace embeddings), with a labeling queue to name each group, plus
  split/merge for corrections.
- **Location clustering** — groups photos by where they were taken, with
  reverse-geocoded place-name suggestions (Kakao for Korea, Yahoo! JAPAN
  for Japan, OpenStreetMap/Nominatim elsewhere — no key required for the
  last one, so this works out of the box).
- **Timeline & groups** — browse photos chronologically, or bundle photos
  into named occasions/events by hand.
- **Autobio** — an AI-drafted diary entry per day (or a narrative spanning
  a date range), editable per segment, exportable as Markdown/text/docx.
  Needs your own Anthropic or OpenAI API key.
- **Six-language UI** (English, Korean, Japanese, Ukrainian, Spanish,
  French), independent of what language your diary text is written in.
- **Bulk seed import** — upload a screenshot of an existing "People &
  Pets" grid to bulk-register known faces before your first real import.

## Setup

### 1. Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Face detection uses InsightFace, which downloads ~300MB of models to
`~/.insightface/models/` on first run.

### 2. Connect a Google Photos account (optional — skip if you're only importing local files)

1. In [Google Cloud Console](https://console.cloud.google.com/), create a
   project and enable the **Google Photos Picker API**.
2. Under APIs & Services → Credentials, create an **OAuth client ID** of
   type **Desktop app**, and download the JSON it gives you.
3. Run the app (see below) and paste that JSON into **Settings → Connect
   your accounts → Google Photos** — easiest, no file editing needed. Or
   save it yourself as `secrets/client_secret.json` (already gitignored).

### 3. Connect an AI provider (optional — only needed for Autobio)

Get an API key from the [Anthropic Console](https://console.anthropic.com/)
or the [OpenAI Platform](https://platform.openai.com/api-keys) — either
works, use whichever you already have. Paste it into **Settings → Connect
your accounts** in the running app, or set it as an environment variable
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) before starting the app.

### 4. Optional: better reverse-geocoding

Place-name suggestions work out of the box via OpenStreetMap. For nicer
results in Korea or Japan, add a free API key:

```bash
export KAKAO_REST_API_KEY=...   # developers.kakao.com — Korea
export YAHOO_JP_CLIENT_ID=...   # developer.yahoo.co.jp — Japan
```

### Run it

```bash
.venv/bin/python -m photo_grouping.web --port 5057
```

Then open `http://127.0.0.1:5057`.

## Running tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Everything is offline and mocked — no live Google/AI calls, no network
required.

## Importing from the command line

For bulk/one-off imports outside the browser UI, `scripts/ingest.py` is
the CLI entry point (safe to re-run — already-imported photos are
skipped):

```bash
.venv/bin/python scripts/ingest.py pick                                    # Picker -> photos + faces
.venv/bin/python scripts/ingest.py backfill-gps --takeout-dir path/to/Takeout
.venv/bin/python scripts/ingest.py cluster-locations
.venv/bin/python scripts/ingest.py review-gps                              # flags GPS outliers for review
```

The Google Photos Picker API doesn't return GPS data through any fetch it
offers, so location data comes from a separate
[Google Takeout](https://takeout.google.com) export (Google Photos only)
matched back to already-imported photos by filename/timestamp.

## Translation management (Crowdin)

UI strings live in `locales/*.json` (`en.json` is the source of truth,
the other five are translations) and can optionally be managed through
[Crowdin](https://crowdin.com) instead of hand-editing each file.

**One-time setup** (your own Crowdin account):

1. Create a free Crowdin project, then note its **Project ID** and
   generate a **Personal Access Token** (Settings → API on each).
2. Install the CLI: `npm install -g @crowdin/cli`
3. Export both as environment variables in your own shell:
   ```bash
   export CROWDIN_PROJECT_ID=your-project-id
   export CROWDIN_PERSONAL_TOKEN=your-token
   ```
   `crowdin.yml` already has the file mapping configured.

**Day to day:**

```bash
crowdin upload sources    # push locales/en.json changes to Crowdin
crowdin download          # pull finished translations back down
```

After a download, restart the app — no code change needed, `i18n.py`
reads the JSON files fresh at startup. Don't hand-edit the five
translated files once this is wired up; the next download will overwrite
them.

## Design notes

- **Face embeddings**: InsightFace (SCRFD detector + ArcFace, 512-d
  embeddings), chosen over dlib after dlib was measured to occasionally
  rank visually-similar people (e.g. a parent and child) closer together
  than genuinely-matching pairs on real test photos. Match threshold is
  tuned against user-assigned labels, not a generic default.
- **Storage**: originals are saved to a local folder (configurable in
  Settings → Photo storage location; defaults to `data/originals/`).
  Local-device imports *copy* the file into that folder rather than
  referencing it in place, so a locally-imported photo exists on disk
  twice — once at its original location, once in the app's storage
  folder. Google-Photos-sourced photos don't have this issue, since the
  copy in the storage folder is the only local copy that ever existed.
- **Known limitation — timezones**: photo timestamps come from EXIF,
  which has no timezone attached. This makes elapsed-time calculations
  slightly unreliable across a timezone change (e.g. during travel),
  which is exactly when large distance jumps are legitimate rather than
  a GPS error — so `review-gps` may flag false positives around trips.
  Not yet fixed.
- **Database**: plain SQLite, stdlib `sqlite3`, no ORM — migrations live
  in `src/photo_grouping/migrations/` and run automatically.

## Open items

- `google_drive`/`dropbox`/`s3` storage backends are stubbed but not
  implemented — `local` and `icloud` (a local folder that happens to
  sync) are the only working options today.
- Per-album (rather than app-wide) storage backend routing isn't
  supported.
- `web.autobio_save` (`POST /autobio/<date>/save`) exists, is tested, but
  isn't called from anywhere in the UI — either needs a "save whole
  entry" button wired up, or is superseded by the per-segment save flow
  and safe to remove.

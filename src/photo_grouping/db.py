"""SQLite connection + migration runner for the Photo Grouping App.

Stdlib-only by design (no ORM, no migration framework) — each user runs
their own local instance of this app, so keeping dependencies minimal keeps
installs simple. See migrations/0001_initial_schema.sql for the schema
itself and the reasoning behind it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with the settings every part of this app relies on."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply any .sql migration files not yet recorded as applied.

    Migrations are plain .sql files named "NNNN_description.sql", applied in
    filename order. Each applied filename is recorded in schema_migrations so
    re-running this is a no-op once everything is applied.

    Returns the list of migration filenames that were newly applied.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )
    conn.commit()

    already_applied = {
        row["version"] for row in conn.execute("SELECT version FROM schema_migrations")
    }

    newly_applied = []
    for migration_file in sorted(migrations_dir.glob("*.sql")):
        version = migration_file.name
        if version in already_applied:
            continue
        sql = migration_file.read_text()
        with conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
            )
        newly_applied.append(version)

    return newly_applied


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Convenience: connect + migrate in one call."""
    conn = connect(db_path)
    migrate(conn)
    return conn


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "photo_grouping.db"
    connection = init_db(target)
    applied = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    print(f"Database at {target} is up to date. Applied migrations:")
    for row in applied:
        print(f"  - {row['version']}")

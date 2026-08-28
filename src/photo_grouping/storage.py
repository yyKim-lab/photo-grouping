"""Pluggable storage adapters for permanently-retained photo originals (§5).

The ingestion pipeline only ever calls `save_original(data, filename)` on
whichever adapter the user's global setting selects — adding a new backend
later (google_drive, dropbox, s3) means writing one class here, not
touching the pipeline itself.

v1 build scope is `local` and `icloud` (both plain filesystem writes, no new
OAuth) — see §5's "Implementation notes". `ICloudStorageAdapter` is a thin
subclass of `LocalStorageAdapter`: iCloud Drive is just a filesystem path
that happens to sync, so the mechanics are identical, matching the existing
Obsidian vault integration pattern in Nexus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageAdapter(Protocol):
    backend_name: str

    def save_original(self, data: bytes, filename: str) -> tuple[str, str]:
        """Persists `data` under `filename`. Returns (backend_name,
        storage_path) — callers store this on
        Photo.original_storage_backend / Photo.original_storage_path."""
        ...


class LocalStorageAdapter:
    """Saves to a configured folder on the local filesystem."""

    backend_name = "local"

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_original(self, data: bytes, filename: str) -> tuple[str, str]:
        path = self._unique_path(filename)
        path.write_bytes(data)
        return self.backend_name, str(path)

    def _unique_path(self, filename: str) -> Path:
        """Google may hand back the same filename for two different photos
        (e.g. camera-default names like IMG_0001.jpg from different
        devices) — never silently overwrite an existing original."""
        path = self.root_dir / filename
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        n = 1
        while True:
            candidate = self.root_dir / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1


class ICloudStorageAdapter(LocalStorageAdapter):
    """Identical mechanics to LocalStorageAdapter — `root_dir` just needs to
    point inside the user's iCloud Drive. No new OAuth needed (§5)."""

    backend_name = "icloud"


def get_adapter(backend_name: str, root_dir: Path) -> StorageAdapter:
    """Global-setting lookup (§5: backend choice is one setting, not
    per-photo, at v1). google_drive/dropbox/s3 raise NotImplementedError
    until their adapters are built — they're offered at onboarding as
    "coming soon" (§5) but aren't v1 build scope."""
    adapters = {
        "local": LocalStorageAdapter,
        "icloud": ICloudStorageAdapter,
    }
    if backend_name in adapters:
        return adapters[backend_name](root_dir)
    if backend_name in ("google_drive", "dropbox", "s3"):
        raise NotImplementedError(
            f"'{backend_name}' storage backend is offered at onboarding as "
            "'coming soon' but has no adapter yet (§5 — v1 build scope is "
            "local + icloud only)."
        )
    raise ValueError(f"Unknown storage backend: {backend_name!r}")

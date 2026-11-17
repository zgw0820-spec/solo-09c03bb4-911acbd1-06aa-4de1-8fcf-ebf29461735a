"""Hot snapshots of the shared training database.

Operators moving the platform to a new host need a restorable backup while
both workers keep training.  A snapshot is produced with SQLite's online
backup API (``Connection.backup``), which gives exactly the guarantees a
migration needs:

* **One commit point.**  The backup API copies the database as of a single
  consistent commit state.  Job progress and weights, checkpoints, events
  and leases in the snapshot therefore always agree with each other - the
  snapshot can never contain half of a batch commit.
* **Non-blocking.**  The database runs in WAL mode, so the backup reader
  does not stop workers from committing batches; pages are copied in chunks
  so writers keep their slots while the copy runs.
* **No half-made files.**  The backup is written to a hidden temporary file
  inside the snapshot directory, verified (``PRAGMA quick_check`` plus the
  expected table set), fsynced, and only then atomically renamed to its
  final name.  Any failure removes the temporary file, and downloads only
  ever see complete snapshots.

The snapshot file is self-contained - no WAL sidecar is needed.  To restore,
place it as ``training.db`` on a fresh Docker volume and start the stack;
workers resume unfinished jobs from the newest committed checkpoint inside
the snapshot, and a lease left behind by the old host simply expires.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import connect

REQUIRED_TABLES = ("jobs", "checkpoints", "job_events", "workers", "leases", "lease_epochs")
SNAPSHOT_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z-]{0,63}$")
MEDIA_TYPE = "application/vnd.sqlite3"

# Copy in chunks rather than all pages at once, so a busy database keeps
# accepting batch commits between backup steps.
_BACKUP_PAGES = 64
_BACKUP_SLEEP_SECONDS = 0.01


class SnapshotError(RuntimeError):
    """The snapshot could not be produced; no partial file was published."""


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: str
    path: Path
    size_bytes: int
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "download_url": f"/snapshots/{self.snapshot_id}/download",
        }


def _run_backup(db_path: Path, target_path: Path) -> None:
    """Copy the live database into ``target_path`` with the online backup API."""
    source = connect(db_path)
    try:
        target = sqlite3.connect(str(target_path))
        try:
            source.backup(target, pages=_BACKUP_PAGES, sleep=_BACKUP_SLEEP_SECONDS)
        finally:
            target.close()
    finally:
        source.close()


def _verify_snapshot(path: Path) -> None:
    """Reject a corrupt or incomplete copy before it is published."""
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("PRAGMA quick_check(1)").fetchone()
        if row is None or row[0] != "ok":
            raise SnapshotError(f"snapshot failed integrity check: {row!r}")
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = sorted(set(REQUIRED_TABLES) - tables)
        if missing:
            raise SnapshotError(f"snapshot is missing tables: {', '.join(missing)}")
    finally:
        conn.close()


def create_snapshot(db_path: Path | str, snapshot_dir: Path | str) -> SnapshotInfo:
    """Take a hot snapshot and publish it atomically in ``snapshot_dir``.

    The temporary file never appears in listings and is removed on any
    failure, so a failed attempt leaves nothing downloadable behind.
    """
    db_path = Path(db_path)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}Z-{uuid.uuid4().hex[:8]}"
    final_path = snapshot_dir / f"{snapshot_id}.db"
    fd, tmp_name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=snapshot_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _run_backup(db_path, tmp_path)
        _verify_snapshot(tmp_path)
        with open(tmp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)  # atomic publish within the same directory
    except BaseException as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError(f"snapshot failed: {exc}") from exc
    return SnapshotInfo(snapshot_id, final_path, final_path.stat().st_size, time.time())


def list_snapshots(snapshot_dir: Path | str) -> list[SnapshotInfo]:
    """All published snapshots, newest first.  Temporary files are hidden."""
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        return []
    infos = []
    for path in snapshot_dir.glob("*.db"):
        if not SNAPSHOT_ID_RE.match(path.stem):
            continue
        stat = path.stat()
        infos.append(SnapshotInfo(path.stem, path, stat.st_size, stat.st_mtime))
    return sorted(infos, key=lambda info: info.snapshot_id, reverse=True)


def find_snapshot(snapshot_dir: Path | str, snapshot_id: str) -> Path | None:
    """Resolve a snapshot id to its file, rejecting ids that escape the dir."""
    if not SNAPSHOT_ID_RE.match(snapshot_id):
        return None
    root = Path(snapshot_dir).resolve()
    path = (root / f"{snapshot_id}.db").resolve()
    if path.parent != root or not path.is_file():
        return None
    return path

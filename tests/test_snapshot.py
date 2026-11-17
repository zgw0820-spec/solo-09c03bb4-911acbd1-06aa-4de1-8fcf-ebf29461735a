"""Hot snapshot API tests: consistency, non-blocking creation, atomic publish."""
from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lease as lm
from app import snapshot, store
from app.config import Settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


def payload(records: int = 16, epochs: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    return {
        "features": rng.normal(size=(records, 3)).tolist(),
        "labels": [i % 2 for i in range(records)],
        "epochs": epochs,
    }


def train_some(client, steps: int = 2, **kwargs):
    """Submit a job and commit ``steps`` batches directly, holding the lease."""
    job = client.post("/jobs", json=payload(**kwargs)).json()
    settings = Settings.from_env()
    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)
    for _ in range(steps):
        store.advance_job(settings.db_path, lease)
    return job, lease


def test_snapshot_roundtrip_download_and_restore_read(client, tmp_path):
    job, _lease = train_some(client)
    settings = Settings.from_env()
    live = client.get(f"/jobs/{job['job_id']}").json()
    live_checkpoints = client.get(f"/jobs/{job['job_id']}/checkpoints").json()

    response = client.post("/snapshots")
    assert response.status_code == 201
    meta = response.json()
    assert meta["snapshot_id"]
    assert meta["size_bytes"] > 0
    assert meta["download_url"] == f"/snapshots/{meta['snapshot_id']}/download"
    assert meta["created_at"] > 0

    listing = client.get("/snapshots").json()
    assert [entry["snapshot_id"] for entry in listing] == [meta["snapshot_id"]]

    download = client.get(meta["download_url"])
    assert download.status_code == 200
    assert "sqlite" in download.headers["content-type"]
    assert "training-" in download.headers["content-disposition"]
    assert download.content[:16] == b"SQLite format 3\x00"  # SQLite magic header

    # The downloaded file is a self-contained, consistent database: progress,
    # weights, checkpoints, events and the lease all come from one commit point.
    copy = tmp_path / "restored.db"
    copy.write_bytes(download.content)
    restored = store.get_job(copy, job["job_id"])  # raises StateConflict if torn
    assert restored["status"] == live["status"]
    assert restored["completed_steps"] == live["completed_steps"] == 2
    assert restored["weights"] == live["weights"]
    assert restored["momentum_velocity"] == live["momentum_velocity"]
    assert restored["loss"] == live["loss"]
    assert store.list_checkpoints(copy, job["job_id"]) == live_checkpoints
    events = store.list_events(copy, job["job_id"])
    assert [event["event"] for event in events] == ["job_created"]
    holder = lm.holder(copy, job["job_id"])
    assert holder is not None and holder[0] == "w1"  # lease travels with the snapshot


def test_snapshot_while_batches_keep_committing(client):
    """Snapshots interleaved with live commits are each internally consistent."""
    settings = Settings.from_env()
    job = client.post("/jobs", json=payload(records=64, epochs=5)).json()  # 40 steps
    lm.register_worker(settings.db_path, "w1")
    lease = lm.acquire(settings.db_path, job["job_id"], "w1", settings.lease_seconds)

    errors: list[BaseException] = []
    committed: list[int] = []
    halt = threading.Event()

    def committer() -> None:
        while not halt.is_set():
            try:
                result = store.advance_job(settings.db_path, lease)
                committed.append(result["step"])
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
            time.sleep(0.002)

    thread = threading.Thread(target=committer)
    thread.start()
    metas = []
    try:
        for _ in range(3):
            response = client.post("/snapshots")
            assert response.status_code == 201
            metas.append(response.json())
    finally:
        halt.set()
        thread.join(timeout=10)

    assert not errors
    assert committed, "workers must keep committing while snapshots are taken"
    for meta in metas:
        path = settings.snapshots_dir() / f"{meta['snapshot_id']}.db"
        # get_job cross-checks progress against the newest checkpoint and
        # raises StateConflict on any torn state.
        state = store.get_job(path, job["job_id"])
        checkpoints = store.list_checkpoints(path, job["job_id"])
        assert [c["step"] for c in checkpoints] == list(range(1, state["completed_steps"] + 1))
        assert state["latest_checkpoint_step"] == state["completed_steps"]


def test_failed_snapshot_publishes_nothing(client, monkeypatch):
    train_some(client)
    settings = Settings.from_env()

    def boom(_db_path, _target):
        raise sqlite3.OperationalError("injected backup failure")

    monkeypatch.setattr(snapshot, "_run_backup", boom)
    response = client.post("/snapshots")
    assert response.status_code == 500
    assert "injected backup failure" in response.json()["detail"]

    # Nothing downloadable and no temporary file left behind.
    assert client.get("/snapshots").json() == []
    snapshot_dir = settings.snapshots_dir()
    leftovers = list(snapshot_dir.iterdir()) if snapshot_dir.exists() else []
    assert leftovers == []


def test_snapshot_download_unknown_or_invalid_id(client):
    train_some(client)
    assert client.get("/snapshots/does-not-exist/download").status_code == 404
    assert client.get("/snapshots/..%2F..%2Ftraining/download").status_code == 404
    assert client.get("/snapshots/bad%20id/download").status_code == 404


def test_swagger_still_lists_all_operations(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    # The pre-existing entries stay available...
    assert "/jobs" in paths
    assert "/jobs/{job_id}/checkpoints" in paths
    assert "/jobs/{job_id}/events" in paths
    assert "/workers" in paths
    # ...and the snapshot operations join them.
    assert "/snapshots" in paths
    assert "/snapshots/{snapshot_id}/download" in paths

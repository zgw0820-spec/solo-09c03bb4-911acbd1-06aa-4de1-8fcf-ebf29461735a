"""Restore drill: a hot snapshot moved to a fresh "volume" resumes training.

Process-level flow, mirroring a host migration:

1. old stack: one job completes, a second job is mid-training with a live
   lease held by worker-a;
2. a hot snapshot is created and downloaded through the API while worker-a
   keeps committing;
3. the old stack dies abruptly (SIGKILL, no cleanup);
4. the snapshot file is placed as ``training.db`` in a brand-new data
   directory (what a fresh Docker volume amounts to) and a new API starts;
5. the new API must serve the original jobs exactly as of the snapshot;
6. a new worker must resume the unfinished job from the newest committed
   checkpoint inside the snapshot - the lease worker-a left behind expires
   instead of stalling the job - and the finished weights must match an
   uninterrupted run bit for bit;
7. the completed job must not be retrained.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from app import store

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 2.0
HEARTBEAT_INTERVAL = 0.3
BATCH_DELAY = 0.25


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE_SECONDS),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT_INTERVAL),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(BATCH_DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


class Server:
    def __init__(self, db: Path, port: int) -> None:
        self.db = db
        self.port = port
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env_for(self.db),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                self.get("/health")
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("API did not start")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict | None = None):
        data = b"{}" if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def download(self, path: str) -> bytes:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as response:
            return response.read()


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for(predicate, timeout: float, interval: float = 0.05, message: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message}")


def wait_completed(server: Server, job_id: str, timeout: float = 120):
    return wait_for(
        lambda: (state := server.get(f"/jobs/{job_id}"))["status"] == "completed" and state,
        timeout=timeout,
        message="job completion",
    )


def stop_worker(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    if hard:
        proc.kill()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(records: int = 40, seed: int = 21, epochs: int = 3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 4)).tolist()
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": epochs}


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """Reference run of the resumable job's input, with no interruption."""
    db = tmp_path_factory.mktemp("reference") / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset())
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)
    return final


def test_restore_from_snapshot_on_fresh_volume(tmp_path, uninterrupted):
    old_db = tmp_path / "old-host" / "training.db"
    snapshot_bytes = b""

    with Server(old_db, free_port()) as old_server:
        done_job = old_server.post("/jobs", dataset(records=16, seed=9, epochs=2))  # 4 steps
        partial_job = old_server.post("/jobs", dataset())  # 15 steps, same input as reference
        worker_a = start_worker(old_db, "worker-a")
        try:
            wait_completed(old_server, done_job["job_id"])
            wait_for(
                lambda: old_server.get(f"/jobs/{partial_job['job_id']}")["completed_steps"] >= 3,
                timeout=60,
                message="partial job progress",
            )
            # Hot snapshot while worker-a keeps training the second job.
            meta = old_server.post("/snapshots")
            snapshot_bytes = old_server.download(meta["download_url"])
        finally:
            stop_worker(worker_a, hard=True)  # the old host goes away abruptly
    assert snapshot_bytes[:16] == b"SQLite format 3\x00"

    # The snapshot is the only thing carried to the new host: a fresh data
    # directory standing in for a brand-new Docker volume.
    new_dir = tmp_path / "new-host"
    new_dir.mkdir()
    new_db = new_dir / "training.db"
    new_db.write_bytes(snapshot_bytes)

    # State as of the snapshot commit point, read from the snapshot itself.
    snap_done = store.get_job(new_db, done_job["job_id"])
    snap_partial = store.get_job(new_db, partial_job["job_id"])
    snap_done_checkpoints = store.list_checkpoints(new_db, done_job["job_id"])
    snap_done_events = store.list_events(new_db, done_job["job_id"])
    assert snap_done["status"] == "completed"
    assert 3 <= snap_partial["completed_steps"] < snap_partial["total_steps"]

    with Server(new_db, free_port()) as new_server:
        # The API on the new host serves the original jobs as of the snapshot.
        restored_done = new_server.get(f"/jobs/{done_job['job_id']}")
        restored_partial = new_server.get(f"/jobs/{partial_job['job_id']}")
        assert restored_done["status"] == "completed"
        assert restored_done["completed_steps"] == snap_done["completed_steps"]
        assert restored_done["final_weights"] == snap_done["final_weights"]
        assert restored_partial["completed_steps"] == snap_partial["completed_steps"]
        assert restored_partial["weights"] == snap_partial["weights"]
        assert restored_partial["status"] in {"queued", "running"}

        # The lease the dead worker-a held is visible in the snapshot...
        leftover = [
            lease
            for lease in new_server.get("/workers")["leases"]
            if lease["job_id"] == partial_job["job_id"]
        ]
        assert leftover and leftover[0]["worker_id"] == "worker-a"

        # ...but it must not stall the job: worker-b takes over once it
        # expires and resumes from the newest committed checkpoint.
        worker_b = start_worker(new_db, "worker-b")
        try:
            final = wait_completed(new_server, partial_job["job_id"])
        finally:
            stop_worker(worker_b)

        events = new_server.get(f"/jobs/{partial_job['job_id']}/events")
        starts = [e for e in events if e["event"] == "training_started"]
        assert starts[-1]["detail"]["worker_id"] == "worker-b"
        assert starts[-1]["detail"]["resumed_from_step"] == snap_partial["completed_steps"]

        # Resumed training is bit-identical to the uninterrupted reference.
        assert final["completed_steps"] == final["total_steps"] == uninterrupted["total_steps"]
        assert final["final_weights"] == uninterrupted["final_weights"]
        assert final["loss"] == uninterrupted["loss"]
        assert final["momentum_velocity"] == uninterrupted["momentum_velocity"]

        checkpoints = new_server.get(f"/jobs/{partial_job['job_id']}/checkpoints")
        assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
        owners = {c["worker_id"] for c in checkpoints}
        assert owners == {"worker-a", "worker-b"}

        # The completed job was not retrained on the new host.
        time.sleep(BATCH_DELAY * 2)  # give any bogus retraining a chance to run
        assert new_server.get(f"/jobs/{done_job['job_id']}/checkpoints") == snap_done_checkpoints
        assert new_server.get(f"/jobs/{done_job['job_id']}/events") == snap_done_events
        done_after = new_server.get(f"/jobs/{done_job['job_id']}")
        assert done_after["status"] == "completed"
        assert done_after["finished_at"] == snap_done["finished_at"]

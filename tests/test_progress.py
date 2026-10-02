"""Progress reporting: client/server round trip, in-thread and isolated
builders, scripts launched with build_env(), reporter snapshots."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from store import (
    Gather,
    LocalExecutor,
    RealizeError,
    Reporter,
    Snapshot,
    build_env,
    derivation,
    output,
    progress,
    realize,
)
from store.progress import ProgressServer, parse_message


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "store"


class Recorder(Reporter):
    def __init__(self):
        self.snapshots: list[Snapshot] = []
        self.closed: Snapshot | None = None

    def update(self, snapshot: Snapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self, snapshot: Snapshot) -> None:
        self.closed = snapshot


def test_parse_message():
    bid, fields = parse_message(b"ID=abc-x\nDONE=3\nTOTAL=10\nSTATUS=hello=world\n")
    assert bid == "abc-x" and fields == {
        "DONE": "3",
        "TOTAL": "10",
        "STATUS": "hello=world",
    }


def test_server_receives_datagrams_from_env_configured_client(monkeypatch):
    got = []
    server = ProgressServer(lambda bid, f: got.append((bid, f)))
    monkeypatch.setenv("STORE_PROGRESS_SOCKET", server.name)
    monkeypatch.setenv("STORE_BUILD_ID", "b1")
    h = progress()
    assert h.active
    h.status("start")
    h.total(4)
    time.sleep(0.15)  # past the coalescing window, so the plain DONE update is sent
    h.set(2)
    deadline = time.time() + 2
    while len(got) < 3 and time.time() < deadline:
        time.sleep(0.02)
    server.close()
    assert got[0][0] == "b1" and got[0][1]["STATUS"] == "start"
    assert float(got[1][1]["TOTAL"]) == 4
    assert float(got[2][1]["DONE"]) == 2


def test_progress_outside_a_build_is_a_noop(monkeypatch):
    monkeypatch.delenv("STORE_PROGRESS_SOCKET", raising=False)
    monkeypatch.delenv("STORE_BUILD_ID", raising=False)
    h = progress()
    assert not h.active
    h.total(3)
    h.set(1)  # must not raise
    env = build_env()
    assert "STORE_PROGRESS_SOCKET" not in env


def test_client_survives_a_dead_server(monkeypatch):
    server = ProgressServer(lambda *a: None)
    name = server.name
    server.close()
    monkeypatch.setenv("STORE_PROGRESS_SOCKET", name)
    monkeypatch.setenv("STORE_BUILD_ID", "b2")
    progress().total(1)  # ECONNREFUSED is swallowed
    progress().set(1)


def test_in_thread_builder_progress_reaches_the_snapshot(store_path: Path):
    rec = Recorder()

    @derivation("steps.txt")
    def steps() -> None:
        p = progress()
        assert p.active
        p.total(5)
        p.status("stepping")
        for k in range(5):
            p.set(k + 1)
            time.sleep(0.05)
        time.sleep(0.3)  # let the last datagram land before we finish
        output().write_text("ok")

    realize(store_path, steps(), executor=LocalExecutor(), reporter=rec)
    st = next(iter(rec.closed.builds.values()))
    assert (
        st.status == "done" and st.total == 5 and st.done == 5 and st.text == "stepping"
    )
    assert st.elapsed is not None and st.elapsed > 0
    assert any(s.count("running") == 1 for s in rec.snapshots)


def test_isolated_builder_progress_reaches_the_snapshot(store_path: Path):
    rec = Recorder()

    @derivation("iso_steps.txt", isolate=True)
    def steps() -> None:
        p = progress()
        assert p.active, "env must carry the socket into the child"
        p.total(3)
        p.set(3)
        p.status("child done")
        time.sleep(0.3)
        output().write_text("ok")

    realize(store_path, steps(), executor=LocalExecutor(), reporter=rec)
    st = next(iter(rec.closed.builds.values()))
    assert (
        st.status == "done"
        and st.done == 3
        and st.total == 3
        and st.text == "child done"
    )


def test_script_launched_with_build_env_reports_progress(
    store_path: Path, tmp_path: Path
):
    rec = Recorder()
    script = tmp_path / "child.py"
    script.write_text(
        "import time\nfrom store import progress\n"
        "p = progress(); assert p.active\n"
        "p.total(2); p.set(2); p.status('from script'); time.sleep(0.3)\n"
    )

    @derivation("via_script.txt")
    def via_script() -> None:
        subprocess.run([sys.executable, str(script)], check=True, env=build_env())
        output().write_text("ok")

    realize(store_path, via_script(), executor=LocalExecutor(), reporter=rec)
    st = next(iter(rec.closed.builds.values()))
    assert st.done == 2 and st.text == "from script"


def test_snapshot_statuses_cover_cached_failed_blocked(store_path: Path):
    @derivation("good.txt")
    def good() -> None:
        output().write_text("g")

    @derivation("bad.txt")
    def bad() -> None:
        raise ValueError("nope")

    @derivation("after.txt")
    def after(b: Path) -> None:
        output().write_text("never")

    realize(store_path, good(), executor=LocalExecutor())  # now cached
    rec = Recorder()
    with pytest.raises(RealizeError):
        realize(
            store_path,
            Gather(good(), after(bad())),
            executor=LocalExecutor(),
            reporter=rec,
        )
    by_name = {b.name: b for b in rec.closed.builds.values()}
    assert by_name["good.txt"].status == "cached"
    assert by_name["bad.txt"].status == "failed" and "nope" in by_name["bad.txt"].error
    assert by_name["after.txt"].status == "blocked"


def test_build_env_inside_a_build_carries_socket_and_id(store_path: Path):
    seen = {}

    @derivation("env.txt")
    def env_probe() -> None:
        e = build_env()
        seen["sock"] = e.get("STORE_PROGRESS_SOCKET")
        seen["id"] = e.get("STORE_BUILD_ID")
        output().write_text("ok")

    d = env_probe()
    realize(store_path, d, executor=LocalExecutor())
    assert (
        seen["sock"].startswith("@store-progress-")
        and seen["id"] == d.hash.hex() + "-env.txt"
    )
    assert (
        os.environ.get("STORE_PROGRESS_SOCKET") is None
    )  # never leaked into the parent env

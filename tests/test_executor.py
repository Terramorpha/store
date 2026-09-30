"""Scheduler / executor behaviour: concurrency, pools, failure semantics,
isolation and build locks."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from store import (
    output,
    Constant,
    LocalExecutor,
    RealizeError,
    derivation,
    expression,
    realize,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "store"


def _sleeper(pool="default", isolate=False):
    @derivation(lambda tag, dt: f"{tag}.txt", pool=pool, isolate=isolate)
    def f(tag: str, dt: float) -> None:
        time.sleep(dt)
        output().write_text(f"{tag} {os.getpid()}")

    return f


def test_independent_derivations_run_concurrently(store_path: Path):
    f = _sleeper()
    t0 = time.time()
    outs = realize(
        store_path,
        [f(f"a{i}", 0.4) for i in range(4)],
        executor=LocalExecutor({"default": 4}),
    )
    assert time.time() - t0 < 1.2  # 4 x 0.4 s serial would be 1.6 s
    assert [o.read_text().split()[0] for o in outs] == [f"a{i}" for i in range(4)]


def test_serial_default_executor(store_path: Path):
    f = _sleeper()
    t0 = time.time()
    realize(store_path, [f(f"s{i}", 0.2) for i in range(3)])
    assert time.time() - t0 >= 0.6


def test_pool_capacity_is_respected(store_path: Path):
    peak = {"train": 0, "eval": 0}
    cur = {"train": 0, "eval": 0}
    lock = threading.Lock()

    def make(pool):
        @derivation(lambda tag: f"{tag}.txt", pool=pool)
        def f(tag: str) -> None:
            with lock:
                cur[pool] += 1
                peak[pool] = max(peak[pool], cur[pool])
            time.sleep(0.15)
            with lock:
                cur[pool] -= 1
            output().write_text(tag)

        return f

    tr, ev = make("train"), make("eval")
    nodes = [tr(f"t{i}") for i in range(6)] + [ev(f"e{i}") for i in range(3)]
    realize(store_path, nodes, executor=LocalExecutor({"train": 2, "eval": 1}))
    assert peak == {"train": 2, "eval": 1}


def test_dependencies_run_before_dependents_and_share_results(store_path: Path):
    order: list[str] = []
    lock = threading.Lock()

    @derivation("base.txt")
    def base() -> None:
        time.sleep(0.1)
        with lock:
            order.append("base")
        output().write_text("3")

    @derivation(lambda b, k: f"child{k}.txt")
    def child(b: Path, k: int) -> None:
        with lock:
            order.append(f"child{k}")
        output().write_text(str(int(b.read_text()) * k))

    b = base()
    outs = realize(
        store_path, [child(b, 1), child(b, 2)], executor=LocalExecutor({"default": 4})
    )
    assert order[0] == "base" and set(order[1:]) == {"child1", "child2"}
    assert [o.read_text() for o in outs] == ["3", "6"]
    assert len(list(store_path.glob("*-base.txt"))) == 1  # built once


def test_failure_blocks_dependents_but_not_independent_work(store_path: Path):
    @derivation("bad.txt")
    def bad() -> None:
        raise ValueError("boom")

    @derivation("after_bad.txt")
    def after_bad(b: Path) -> None:
        output().write_text("never")

    @derivation("good.txt")
    def good() -> None:
        output().write_text("ok")

    with pytest.raises(RealizeError) as ei:
        realize(
            store_path,
            [after_bad(bad()), good()],
            executor=LocalExecutor({"default": 2}),
        )
    err = ei.value
    assert list(err.failed) == [bad().hash.hex() + "-bad.txt"]
    assert isinstance(err.failed[bad().hash.hex() + "-bad.txt"], ValueError)
    assert err.blocked == [after_bad(bad()).hash.hex() + "-after_bad.txt"]
    assert (store_path / (good().hash.hex() + "-good.txt")).read_text() == "ok"
    assert not list(store_path.glob("*.tmp-*")) and not list(store_path.glob("*.lock"))


def test_fail_fast_raises_immediately(store_path: Path):
    @derivation("bad.txt")
    def bad() -> None:
        raise RuntimeError("boom")

    @derivation("slow.txt")
    def slow() -> None:
        time.sleep(2.0)
        output().write_text("slow")

    t0 = time.time()
    with pytest.raises(RealizeError):
        realize(
            store_path,
            [bad(), slow()],
            executor=LocalExecutor({"default": 2}),
            fail_fast=True,
        )
    assert time.time() - t0 < 1.5


def test_expression_is_evaluated_once_per_realization(store_path: Path):
    calls: list[int] = []

    @expression()
    def shared() -> int:
        calls.append(1)
        return 5

    @derivation(lambda v, k: f"use{k}.txt")
    def use(v: int, k: int) -> None:
        output().write_text(str(v * k))

    e = shared()
    outs = realize(
        store_path, [use(e, 1), use(e, 2), e], executor=LocalExecutor({"default": 2})
    )
    assert calls == [1]
    assert [outs[0].read_text(), outs[1].read_text(), outs[2]] == ["5", "10", 5]


def test_isolated_builder_runs_in_another_interpreter(store_path: Path):
    f = _sleeper(isolate=True)
    out = realize(store_path, f("iso", 0.0))
    tag, pid = out.read_text().split()
    assert tag == "iso" and int(pid) != os.getpid()


def test_isolated_builder_failure_is_reported(store_path: Path):
    @derivation("iso_bad.txt", isolate=True)
    def bad() -> None:
        raise ValueError("child boom")

    with pytest.raises(RealizeError) as ei:
        realize(store_path, bad())
    (exc,) = ei.value.failed.values()
    assert "child boom" in str(exc)


def test_live_lock_makes_second_realizer_wait_for_the_output(store_path: Path):
    @derivation("shared.txt")
    def shared() -> None:
        output().write_text("mine")

    d = shared()
    store_path.mkdir()
    out = store_path / (d.hash.hex() + "-shared.txt")
    lock = out.with_name(out.name + ".lock")
    lock.write_text("{}")  # a fresh lock: someone else is building

    def finish_other_build():
        time.sleep(0.6)
        out.write_text("theirs")
        lock.unlink()

    threading.Thread(target=finish_other_build).start()
    t0 = time.time()
    res = realize(store_path, d, executor=LocalExecutor(poll=0.1))
    assert time.time() - t0 >= 0.5
    assert res.read_text() == "theirs"  # we did not rebuild


def test_stale_lock_is_taken_over(store_path: Path):
    @derivation("stale.txt")
    def stale() -> None:
        output().write_text("rebuilt")

    d = stale()
    store_path.mkdir()
    out = store_path / (d.hash.hex() + "-stale.txt")
    lock = out.with_name(out.name + ".lock")
    lock.write_text("{}")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    res = realize(store_path, d, executor=LocalExecutor(lock_stale_after=60, poll=0.05))
    assert res.read_text() == "rebuilt" and not lock.exists()


def test_nested_realize_inside_a_builder(store_path: Path):
    @derivation("inner.txt")
    def inner() -> None:
        output().write_text("inner")

    @derivation("outer.txt")
    def outer() -> None:
        p = realize(store_path, inner())  # builders may realize on their own
        output().write_text(p.read_text() + "+outer")

    assert (
        realize(store_path, outer(), executor=LocalExecutor({"default": 2})).read_text()
        == "inner+outer"
    )


def test_list_of_roots_returns_values_in_order(store_path: Path):
    @derivation("x.txt")
    def x() -> None:
        output().write_text("x")

    res = realize(store_path, [Constant(1), x(), Constant("c")])
    assert res[0] == 1 and res[1].read_text() == "x" and res[2] == "c"


def test_legacy_output_alias_works_including_isolated(store_path: Path):
    from store import OUTPUT

    @derivation("legacy.txt", isolate=True)
    def legacy() -> None:
        OUTPUT.get().write_text("legacy")

    assert realize(store_path, legacy()).read_text() == "legacy"
    with pytest.raises(RuntimeError, match="outside of a builder"):
        OUTPUT.get()

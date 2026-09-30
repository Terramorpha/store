"""The executor layer: LocalExecutor (pools of build slots), one build of a
derivation (_Build: temp output, isolation, abort), and the build lock that
coordinates realizers across threads and processes.
"""

import inspect
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from store.graph import _OUTPUT, Derivation, Realizable

logger = logging.getLogger(__name__)


class LocalExecutor:
    """Runs builders on this machine, concurrently, with a slot budget per
    pool. ``pools`` maps a pool name to how many of its builds may run at
    once; a pool not listed gets ``default_pool_size`` slots. ``LocalExecutor()``
    with no arguments is the serial executor (one slot per pool).

    Builds run on daemon worker threads. A builder that spends its time in a
    subprocess (EnergyPlus, a training script) releases the GIL, so threads
    are enough; a builder that must not share interpreter state, or that does
    heavy in-process Python work, is declared ``isolate=True`` on its
    derivation and is run in a fresh interpreter (see ``_run_isolated``).

    ``lock_stale_after`` (seconds) is how old a build lock's heartbeat may be
    before the build behind it is presumed dead and the lock is taken over.
    """

    def __init__(
        self,
        pools: dict[str, int] | None = None,
        *,
        default_pool_size: int = 1,
        lock_stale_after: float = 300.0,
        heartbeat: float = 30.0,
        poll: float = 1.0,
    ):
        self.pools = dict(pools or {})
        self.default_pool_size = default_pool_size
        self.lock_stale_after = lock_stale_after
        self.heartbeat = heartbeat
        self.poll = poll

    def capacity(self, pool: str) -> int:
        return self.pools.get(pool, self.default_pool_size)


def _node_key(r: "Realizable") -> tuple[str, bytes]:
    return (type(r).__name__, r.hash)


def _label(r: "Realizable") -> str:
    if isinstance(r, Derivation):
        return r.hash.hex() + "-" + r.name
    return "expr-" + r.hash.hex()


def _remove_path(p: Path) -> None:
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except Exception:
        pass


class _Build:
    """One derivation being built: owns its lock, temp path and (for
    ``isolate``) the child process, so the scheduler can abort it cleanly."""

    def __init__(
        self,
        store_path: Path,
        derivation: Derivation,
        realized_deps: list[Any],
        executor: LocalExecutor,
    ):
        self.store_path = store_path
        self.derivation = derivation
        self.realized_deps = realized_deps
        self.executor = executor
        self.output_path = store_path / _label(derivation)
        self.tmp_output_path = (
            store_path / f"{_label(derivation)}.tmp-{os.getpid()}-{uuid4().hex}"
        )
        self.lock: _BuildLock | None = None
        self.proc: subprocess.Popen | None = None
        self.aborted = threading.Event()

    def run(self) -> Path:
        """Build unless the output exists or another realizer finishes it
        first; returns the output path."""
        if self.output_path.exists():
            return self.output_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.lock = _BuildLock(self.output_path, self.executor, self.aborted)
        if not self.lock.acquire():
            return self.output_path
        try:
            logger.info(f"building {self.derivation.name}")
            self._call_builder()
            if self.aborted.is_set():
                raise _Aborted()
            if not self.tmp_output_path.exists():
                file_path = inspect.getsourcefile(self.derivation.builder)
                _lines, start_line = inspect.getsourcelines(self.derivation.builder)
                raise Exception(
                    f"derivation {self.derivation.name} did not produce an output "
                    f"at output(). Perhaps make the builder at "
                    f"{file_path}:{start_line} not silently fail?"
                )
            if self.output_path.exists():  # someone else finished first
                _remove_path(self.tmp_output_path)
                return self.output_path
            try:
                self.tmp_output_path.rename(self.output_path)
            except FileExistsError:
                _remove_path(self.tmp_output_path)
        except BaseException:
            _remove_path(self.tmp_output_path)
            raise
        finally:
            self.lock.release()
        return self.output_path

    def _call_builder(self) -> None:
        if self.derivation.isolate:
            self._run_isolated()
            return
        token = _OUTPUT.set(self.tmp_output_path)
        try:
            self.derivation.builder(self.realized_deps)
        finally:
            _OUTPUT.reset(token)

    def _run_isolated(self) -> None:
        """Run the builder in a child interpreter: the closure, its realized
        dependencies and the output path are cloudpickled to a file and
        ``python -m store._isolated <file>`` binds output() and calls it. The
        child inherits this process's environment and cwd."""
        import cloudpickle

        with tempfile.NamedTemporaryFile(
            "wb", prefix="store-isolated-", suffix=".pkl", delete=False
        ) as f:
            payload = f.name
            cloudpickle.dump(
                (self.derivation.builder, self.realized_deps, self.tmp_output_path), f
            )
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "store._isolated", payload],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _out, err = self.proc.communicate()
        finally:
            Path(payload).unlink(missing_ok=True)
        if self.aborted.is_set():
            return
        if self.proc.returncode != 0:
            raise RuntimeError(
                f"isolated build of {self.derivation.name} exited with "
                f"{self.proc.returncode}:\n{err[-4000:]}"
            )

    def abort(self) -> None:
        """Stop this build now (preemption / fail-fast): kill an isolated child,
        release the lock, drop the temp output. An in-thread builder cannot be
        interrupted; its daemon thread simply dies with the process and its
        result is ignored."""
        self.aborted.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.lock is not None:
            self.lock.release()
        _remove_path(self.tmp_output_path)


class _Aborted(Exception):
    """Internal: a build finished after it was aborted; its result is dropped."""


class _BuildLock:
    """``<output>.lock`` with a heartbeat: tells other realizers (threads or
    processes, on the same filesystem) that this output is being built, so
    they wait for it instead of building it again. A lock whose heartbeat is
    older than ``lock_stale_after`` belongs to a dead build and is taken over."""

    def __init__(
        self, output_path: Path, executor: LocalExecutor, aborted: threading.Event
    ):
        self.path = output_path.with_name(output_path.name + ".lock")
        self.output_path = output_path
        self.executor = executor
        self.aborted = aborted
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._held = False

    def acquire(self) -> bool:
        """Return True when we hold the lock, False when the output appeared
        while we were waiting for someone else's build (or we were aborted)."""
        while not self.aborted.is_set():
            if self.output_path.exists():
                return False
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.executor.lock_stale_after:
                    logger.warning(
                        f"taking over stale lock {self.path.name} "
                        f"(heartbeat {age:.0f}s old)"
                    )
                    self.path.unlink(missing_ok=True)
                    continue
                time.sleep(self.executor.poll)
                continue
            with os.fdopen(fd, "w") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "started": time.time(),
                    },
                    f,
                )
            self._held = True
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()
            return True
        return False

    def _beat(self) -> None:
        while not self._stop.wait(self.executor.heartbeat):
            try:
                os.utime(self.path, None)
            except FileNotFoundError:
                return

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        self._stop.set()
        self.path.unlink(missing_ok=True)

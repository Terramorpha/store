"""The scheduler: realize() runs the DAG under one node in dependency order,
concurrently within the executor's pools, with preemption and failure
semantics (RealizeError).
"""

import logging
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, overload

from store.executor import LocalExecutor, _Aborted, _Build, _label, _node_key
from store.graph import Derivation, Expression, Realizable, Result
from store.progress import BuildState, ProgressServer, Reporter, Snapshot

logger = logging.getLogger(__name__)


class RealizeError(Exception):
    """Raised at the end of a realization in which at least one derivation
    failed. ``failed`` maps the node's ``<hash>-<name>`` to the exception;
    ``blocked`` lists the nodes that could not run because a dependency
    failed. Everything independent of the failures was still built."""

    def __init__(self, failed: dict[str, BaseException], blocked: list[str]):
        self.failed = failed
        self.blocked = blocked
        lines = [f"{len(failed)} derivation(s) failed, {len(blocked)} blocked"]
        for key, exc in failed.items():
            lines.append(f"  FAILED  {key}: {type(exc).__name__}: {exc}")
        for key in blocked:
            lines.append(f"  BLOCKED {key}")
        super().__init__("\n".join(lines))


class _Scheduler:
    """One realization: the DAG under the requested root, run in dependency
    order with as much concurrency as the executor's pools allow.

    Builds run on daemon threads owned by this scheduler (never on the
    calling thread), and results come back through a queue the scheduler
    blocks on; a KeyboardInterrupt on the calling thread interrupts that
    wait and is treated as preemption: every running build is
    aborted (isolated children killed, locks released, temp outputs removed)
    and the interrupt propagates. That is distinct from a build failing,
    which only blocks that build's dependents."""

    def __init__(
        self,
        store_path: Path,
        executor: LocalExecutor,
        fail_fast: bool,
        reporter: Reporter | None = None,
    ):
        self.store_path = store_path
        self.executor = executor
        self.fail_fast = fail_fast
        self.reporter = reporter or Reporter()
        self.snapshot = Snapshot(started_at=time.time())
        self._state_lock = threading.Lock()
        self._last_notify = 0.0
        self.server = ProgressServer(self._on_progress)
        self.nodes: dict[tuple[str, bytes], Realizable] = {}
        self.deps: dict[tuple[str, bytes], list[tuple[str, bytes]]] = {}
        self.dependents: dict[tuple[str, bytes], set[tuple[str, bytes]]] = {}
        self.results: dict[tuple[str, bytes], Any] = {}
        self.failed: dict[tuple[str, bytes], BaseException] = {}
        self.blocked: set[tuple[str, bytes]] = set()
        self.running: dict[tuple[str, bytes], _Build] = {}
        self.done_queue: queue.Queue[
            tuple[tuple[str, bytes], Any, BaseException | None]
        ] = queue.Queue()

    def _collect(self, r: Realizable) -> None:
        k = _node_key(r)
        if k in self.nodes:
            return
        if not isinstance(r, (Derivation, Expression)):
            raise ValueError(f"Unknown realizable type: {type(r)}")
        self.nodes[k] = r
        if isinstance(r, Derivation):
            cap = self.executor.capacity(r.pool)
            if r.slots > cap:
                raise ValueError(
                    f"{r.name} needs {r.slots} slots of pool {r.pool!r}, "
                    f"which has {cap}"
                )
            self.snapshot.builds[_label(r)] = BuildState(
                label=_label(r), name=r.name, pool=r.pool, slots=r.slots
            )
        self.deps[k] = [_node_key(d) for d in r.dependencies]
        self.dependents.setdefault(k, set())
        for d in r.dependencies:
            self._collect(d)
            self.dependents[_node_key(d)].add(k)

    # --- progress / state model ------------------------------------------
    def _set_status(self, label: str, status: str, **fields: Any) -> None:
        with self._state_lock:
            st = self.snapshot.builds.get(label)
            if st is None:
                return
            st.status = status
            for name, value in fields.items():
                setattr(st, name, value)
        self._notify()

    def _on_progress(self, build_id: str, fields: dict[str, str]) -> None:
        with self._state_lock:
            st = self.snapshot.builds.get(build_id)
            if st is None:
                return
            try:
                if "DONE" in fields:
                    st.done = float(fields["DONE"])
                if "TOTAL" in fields:
                    st.total = float(fields["TOTAL"])
            except ValueError:
                pass
            if "STATUS" in fields:
                st.text = fields["STATUS"]
        self._notify()

    def _copy_snapshot(self) -> Snapshot:
        """Reporters get a copy: a snapshot is a value, not a live view."""
        with self._state_lock:
            return Snapshot(
                started_at=self.snapshot.started_at,
                builds={k: replace(v) for k, v in self.snapshot.builds.items()},
            )

    def _notify(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_notify < 0.1:
            return
        self._last_notify = now
        try:
            self.reporter.update(self._copy_snapshot())
        except Exception:  # noqa: BLE001 - a reporter bug must not break the build
            logger.exception("reporter.update failed")

    def _block(self, k: tuple[str, bytes]) -> None:
        for dep_k in self.dependents[k]:
            if dep_k not in self.blocked and dep_k not in self.results:
                self.blocked.add(dep_k)
                if isinstance(self.nodes[dep_k], Derivation):
                    self._set_status(_label(self.nodes[dep_k]), "blocked")
                self._block(dep_k)

    def _fail(self, k: tuple[str, bytes], exc: BaseException) -> None:
        logger.error(f"failed: {_label(self.nodes[k])}: {exc!r}")
        self.failed[k] = exc
        if isinstance(self.nodes[k], Derivation):
            self._set_status(
                _label(self.nodes[k]),
                "failed",
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
            )
        self._block(k)

    def _start(self, k: tuple[str, bytes], build: _Build) -> None:
        def work() -> None:
            try:
                result = build.run()
            except BaseException as exc:  # noqa: BLE001 - reported through the queue
                self.done_queue.put((k, None, exc))
            else:
                self.done_queue.put((k, result, None))

        self.running[k] = build
        self._set_status(build.label, "running", started_at=time.time())
        threading.Thread(
            target=work, name=f"store-build-{build.derivation.name}", daemon=True
        ).start()

    def _abort_all(self) -> None:
        for build in list(self.running.values()):
            build.abort()
        self.running.clear()

    def _error(self) -> RealizeError:
        return RealizeError(
            {_label(self.nodes[k]): e for k, e in self.failed.items()},
            sorted(_label(self.nodes[b]) for b in self.blocked),
        )

    def run(self, root: Realizable) -> Any:
        self._collect(root)
        pending = set(self.nodes)
        in_use: dict[str, int] = {}
        try:
            while pending or self.running:
                self._submit_ready(pending, in_use)
                if not self.running:
                    if pending:  # a cycle or an internal bug: nothing can become ready
                        raise RuntimeError(
                            f"realize: {len(pending)} node(s) can never become ready"
                        )
                    break
                # A blocking get(): on CPython/POSIX a lock wait in the main
                # thread is interrupted by signals, so Ctrl-C lands here at once.
                k, result, exc = self.done_queue.get()
                build = self.running.pop(k, None)
                if build is None:  # finished after an abort: ignore
                    continue
                in_use[self.nodes[k].pool] -= self.nodes[k].slots
                if exc is None:
                    self.results[k] = result
                    self._set_status(build.label, "done", finished_at=time.time())
                elif isinstance(exc, _Aborted):
                    continue
                else:
                    self._fail(k, exc)
                    if self.fail_fast:
                        self._abort_all()
                        raise self._error()
        except BaseException:
            # Preemption (KeyboardInterrupt, SystemExit, ...) or fail-fast:
            # stop everything that is running before propagating.
            self._abort_all()
            raise
        finally:
            self.server.close()
            try:
                self.reporter.close(self._copy_snapshot())
            except Exception:  # noqa: BLE001
                logger.exception("reporter.close failed")
        if self.failed:
            raise self._error()
        return self.results[_node_key(root)]

    def _submit_ready(self, pending: set, in_use: dict[str, int]) -> None:
        """Start every derivation whose dependencies are done (within pool
        capacity) and evaluate every ready expression inline."""
        progressed = True
        while progressed:
            progressed = False
            for k in sorted(pending, key=lambda k: self.nodes[k].hash):
                if k in self.blocked:
                    pending.discard(k)
                    progressed = True
                    continue
                if any(d not in self.results for d in self.deps[k]):
                    continue
                node = self.nodes[k]
                realized = [self.results[d] for d in self.deps[k]]
                if isinstance(node, Expression):
                    # Expressions are cheap glue: evaluate inline, memoised
                    # for this run.
                    pending.discard(k)
                    try:
                        self.results[k] = node.builder(realized)
                    except Exception as exc:  # noqa: BLE001 - recorded as a failure
                        self._fail(k, exc)
                    progressed = True
                    continue
                out = self.store_path / _label(node)
                if out.exists():
                    pending.discard(k)
                    self.results[k] = out
                    self._set_status(_label(node), "cached")
                    progressed = True
                    continue
                cap = self.executor.capacity(node.pool)
                if in_use.get(node.pool, 0) + node.slots > cap:
                    continue
                in_use[node.pool] = in_use.get(node.pool, 0) + node.slots
                pending.discard(k)
                label = _label(node)
                self._start(
                    k,
                    _Build(
                        self.store_path,
                        node,
                        realized,
                        self.executor,
                        progress_socket=self.server.name,
                        on_status=lambda status, label=label: self._set_status(
                            label, status
                        ),
                    ),
                )
                progressed = True


@overload
def realize(
    store_path: Path,
    expression: Expression[Result],
    *,
    executor: LocalExecutor,
    fail_fast: bool = False,
    reporter: Reporter | None = None,
) -> Result: ...
@overload
def realize(
    store_path: Path,
    derivation: Derivation,
    *,
    executor: LocalExecutor,
    fail_fast: bool = False,
    reporter: Reporter | None = None,
) -> Path: ...
def realize(store_path, realizable, *, executor, fail_fast=False, reporter=None):
    """Realize one node: a derivation (returns its Path) or an expression
    (returns its value), building whatever is missing under it.

    ``executor`` says how builds run and is always given explicitly:
    ``LocalExecutor()`` is serial, ``LocalExecutor({"pool": n, ...})`` runs
    independent dependencies concurrently within each pool's slot budget. To
    build several unrelated nodes together, make them the dependencies of one
    node, e.g. ``Gather(a, b, c)``.

    A failed derivation blocks its dependents but nothing else; when
    everything runnable has run, :class:`RealizeError` reports the failures
    and the blocked nodes. ``fail_fast=True`` aborts the running builds and
    raises at the first failure. A KeyboardInterrupt on the calling thread is
    preemption: running builds are aborted (isolated children killed, locks
    released, temp outputs removed) and the interrupt propagates at once.

    ``reporter`` receives a :class:`~store.progress.Snapshot` of every
    build's state whenever it changes (``store.ui.RichReporter`` renders a
    live terminal table); builders report their own progress through
    :func:`store.progress`.
    """
    if not isinstance(realizable, (Derivation, Expression)):
        raise TypeError(
            "realize() takes a Derivation or an Expression, not "
            f"{type(realizable).__name__}; wrap several nodes in Gather(...)"
        )
    if not isinstance(executor, LocalExecutor):
        raise TypeError(
            f"executor must be a LocalExecutor, not {type(executor).__name__}"
        )
    scheduler = _Scheduler(Path(store_path), executor, fail_fast, reporter)
    return scheduler.run(realizable)

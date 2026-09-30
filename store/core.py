"""Core of the store: derivations, expressions, realization, and the stock
concrete derivations (downloads, git clones, local files, archives).

See the package README for the model. Everything public is re-exported from
``store``.
"""

import hashlib
import inspect
import json
import logging
import os
import pickle
import queue
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Generic, TypeVar, overload
from uuid import uuid4

import git
import requests
from rich.progress import DownloadColumn, Progress, TransferSpeedColumn

logger = logging.getLogger(__name__)

# Where the builder currently being run must write its output. Bound by the
# realizer around each builder call (per thread / per isolated child); builders
# read it through output(), never through the variable itself.
_OUTPUT: ContextVar[Path] = ContextVar("OUTPUT")


def output() -> Path:
    """The path the running builder must write its output to (a file or a
    directory; the builder decides). Only meaningful inside a builder."""
    try:
        return _OUTPUT.get()
    except LookupError:
        raise RuntimeError("store.output() called outside of a builder") from None


class _OutputAlias:
    """Compatibility shim for builders written as ``OUTPUT.get()``: a plain,
    stateless singleton whose ``get()`` is :func:`output`. Being an ordinary
    object it pickles like any other global, so legacy builders work under
    ``isolate=True`` too."""

    def get(self) -> Path:
        return output()

    def __repr__(self) -> str:
        return "OUTPUT"


OUTPUT = _OutputAlias()


DEFAULT_POOL = "default"


@dataclass(frozen=True)
class Derivation:
    name: str
    hash: bytes
    dependencies: list["Realizable"]
    # Builder receives only the realized dependencies; the output path is read
    # from the OUTPUT ContextVar.
    builder: Callable[[list[Any]], None]
    # Scheduling: which executor pool this build occupies a slot of, and whether
    # the builder must run in a fresh interpreter (crash / global-state isolation).
    pool: str = DEFAULT_POOL
    isolate: bool = False

    def __post_init__(self):
        assert Path(self.name).name == self.name, (
            "name of derivation can't contain a slash"
        )


Result = TypeVar("Result")


@dataclass(frozen=True)
class Expression(Generic[Result]):
    hash: bytes
    dependencies: list["Realizable"]
    builder: Callable[[list[Any]], Result]


Realizable = Derivation | Expression


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

    def __init__(self, store_path: Path, executor: LocalExecutor, fail_fast: bool):
        self.store_path = store_path
        self.executor = executor
        self.fail_fast = fail_fast
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
        self.deps[k] = [_node_key(d) for d in r.dependencies]
        self.dependents.setdefault(k, set())
        for d in r.dependencies:
            self._collect(d)
            self.dependents[_node_key(d)].add(k)

    def _block(self, k: tuple[str, bytes]) -> None:
        for dep_k in self.dependents[k]:
            if dep_k not in self.blocked and dep_k not in self.results:
                self.blocked.add(dep_k)
                self._block(dep_k)

    def _fail(self, k: tuple[str, bytes], exc: BaseException) -> None:
        logger.error(f"failed: {_label(self.nodes[k])}: {exc!r}")
        self.failed[k] = exc
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
                in_use[self.nodes[k].pool] -= 1
                if exc is None:
                    self.results[k] = result
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
                    progressed = True
                    continue
                cap = self.executor.capacity(node.pool)
                if in_use.get(node.pool, 0) >= cap:
                    continue
                in_use[node.pool] = in_use.get(node.pool, 0) + 1
                pending.discard(k)
                self._start(k, _Build(self.store_path, node, realized, self.executor))
                progressed = True


@overload
def realize(
    store_path: Path,
    expression: Expression[Result],
    *,
    executor: LocalExecutor,
    fail_fast: bool = False,
) -> Result: ...
@overload
def realize(
    store_path: Path,
    derivation: Derivation,
    *,
    executor: LocalExecutor,
    fail_fast: bool = False,
) -> Path: ...
def realize(store_path, realizable, *, executor, fail_fast=False):
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
    scheduler = _Scheduler(Path(store_path), executor, fail_fast)
    return scheduler.run(realizable)


def compute_hash(
    func, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bytes:
    """Compute a content hash for an expression/derivation input.

    - Includes the function name.
    - For Realizable args/kwargs, uses their .hash directly.
    - For everything else, uses pickle to serialize and hash the bytes.
    """
    hasher = hashlib.blake2b(digest_size=32)

    if inspect.isfunction(func):
        hasher.update(func.__name__.encode("utf-8"))
        hasher.update(inspect.getsource(func).encode("utf-8"))

    hasher.update(name.encode("utf-8"))

    # Positional args
    for arg in args:
        if isinstance(arg, (Derivation, Expression)):
            hasher.update(arg.hash)
        else:
            payload = pickle.dumps(arg)
            hasher.update(payload)

    # Keyword args (preserve call order)
    for key, value in kwargs.items():
        hasher.update(str(key).encode("utf-8"))
        if isinstance(value, (Derivation, Expression)):
            hasher.update(value.hash)
        else:
            payload = pickle.dumps(value)
            hasher.update(payload)

    return hasher.digest()


def _capture_dependencies_and_builder(func: Callable[..., Any], *args, **kwargs):
    """Internal helper used by the decorators to:
    - capture Realizable dependencies in call order
    - build placeholder structures to reconstruct args/kwargs
    - return (dependencies, builder)
    The resulting builder takes a single argument: the list of realized dependencies
    in the same order they were captured.
    """
    dependencies: list[Realizable] = []

    # Placeholders for reconstruction
    pos_placeholders: list[Any | None] = []
    for a in args:
        if isinstance(a, (Derivation, Expression)):
            dependencies.append(a)
            pos_placeholders.append(None)
        else:
            pos_placeholders.append(a)

    kw_placeholders: dict[str, Any | None] = {}
    for k, v in kwargs.items():
        if isinstance(v, (Derivation, Expression)):
            dependencies.append(v)
            kw_placeholders[k] = None
        else:
            kw_placeholders[k] = v

    @wraps(func)
    def builder(realized_deps: list[Any]) -> Any:
        dep_iter = iter(realized_deps)

        final_args: list[Any] = []
        for val in pos_placeholders:
            if val is None:
                final_args.append(next(dep_iter))
            else:
                final_args.append(val)

        final_kwargs: dict[str, Any] = {}
        for k, v in kw_placeholders.items():
            if v is None:
                final_kwargs[k] = next(dep_iter)
            else:
                final_kwargs[k] = v

        return func(*final_args, **final_kwargs)

    return dependencies, builder


def expression() -> Callable[
    [Callable[..., Result]], Callable[..., Expression[Result]]
]:
    """Decorator: calling the wrapped function returns an Expression node."""

    def decorator(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            dependencies, builder = _capture_dependencies_and_builder(
                func, *args, **kwargs
            )
            return Expression(
                hash=compute_hash(func, "expr", args, kwargs),
                dependencies=dependencies,
                builder=builder,
            )

        return wrapper

    return decorator


def derivation(
    name: str | Callable,
    *,
    pool: str = DEFAULT_POOL,
    isolate: bool = False,
) -> Callable[[Callable[..., None]], Callable[..., Derivation]]:
    """Decorator: calling the wrapped function returns a Derivation node.

    ``pool`` names the executor pool whose slot the build occupies (see
    :class:`LocalExecutor`); ``isolate=True`` runs the builder in a fresh
    interpreter.

    The constructed builder takes only the realized dependency values. During
    realization, the output path is made available via the ContextVar
    ``output()`` called inside the function body.
    """

    def compute_name(args, kwargs):
        if isinstance(name, str):
            return name
        elif isinstance(name, Callable):
            return name(*args, **kwargs)

    def decorator(func) -> Callable[..., Derivation]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Derivation:
            dependencies, builder = _capture_dependencies_and_builder(
                func, *args, **kwargs
            )

            der_name = compute_name(args, kwargs)
            return Derivation(
                name=der_name,
                hash=compute_hash(func, der_name, args, kwargs),
                dependencies=dependencies,
                builder=builder,  # returns None; writes to output()
                pool=pool,
                isolate=isolate,
            )

        return wrapper

    return decorator


# concrete implementations


def DownloadFile(
    filename: str,
    url: str,
    hash: bytes | None,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    if hash is not None:
        derivation_hash = hash
    else:
        h = hashlib.sha256()
        h.update(url.encode("utf-8"))
        derivation_hash = h.digest()

    def builder(_):
        out = output()

        # Get progress object and create a task

        with Progress(
            *Progress.get_default_columns(),
            DownloadColumn(),
            TransferSpeedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading {filename}", total=None)

            hasher = hasher_factory()
            # Verify TLS certificates by default.
            #
            # If your environment requires a custom CA bundle, set one of:
            # - B2B_CA_BUNDLE=/path/to/ca-bundle.pem
            # - REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
            # - SSL_CERT_FILE=/path/to/ca-bundle.pem
            #
            # To explicitly opt-out (NOT recommended), set:
            # - B2B_INSECURE_SSL=1
            verify: bool | str = True
            if os.environ.get("B2B_INSECURE_SSL", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                verify = False
            else:
                ca_bundle = (
                    os.environ.get("B2B_CA_BUNDLE")
                    or os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("SSL_CERT_FILE")
                )
                if ca_bundle:
                    verify = ca_bundle

            response = requests.get(url, stream=True, verify=verify)
            response.raise_for_status()

            # Update task with actual file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                progress.update(task, total=total_size)

            block_size = 1 << 16

            with tempfile.NamedTemporaryFile("wb", delete=False) as outfile:
                for data in response.iter_content(block_size):
                    hasher.update(data)
                    outfile.write(data)
                    progress.update(task, advance=len(data))

            outfile.close()
            h = hasher.digest()
            if hash is not None and hash != h:
                raise Exception(
                    f"Hash of download {filename} is wrong. "
                    f"Expected: {hash.hex()}, actual: {h.hex()} "
                    f"(computed using {hasher})"
                )
            shutil.move(outfile.name, out)

    return Derivation(filename, derivation_hash, [], builder)


def ExtractTarball(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".tar.gz"))
    def inner(input: Path):
        dst = output()

        with tarfile.open(input, "r:gz") as tar:
            tar.extractall(path=dst)

    return inner(input_der)


def ExtractZip(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".zip"))
    def inner(input: Path):
        dst = output()

        dst.mkdir()

        with zipfile.ZipFile(input, "r") as zip_ref:
            file_list = zip_ref.infolist()

            for file_info in file_list:
                zip_ref.extract(file_info, dst)

    return inner(input_der)


def ExtractFromZip(zip_file: Realizable, filename: str) -> Derivation:
    @derivation(Path(filename).name)
    def inner(input: Path, name: str):
        dst = output()
        with zipfile.ZipFile(input, "r") as zip_ref:
            with zip_ref.open(name) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)

    return inner(zip_file, filename)


def hash_directory_tree(hasher, dir: Path):
    # Get all files and sort them for deterministic ordering
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()
        for file in files:
            file_paths.append(os.path.join(root, file))

    # Sort all file paths to ensure deterministic order
    file_paths.sort()

    # Hash each file's content
    for file_path in file_paths:
        # Include the relative path in the hash for structure integrity
        rel_path = os.path.relpath(file_path, dir)
        hasher.update(rel_path.encode("utf-8"))

        # Hash the file content
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)


def GitClone(
    filename: str,
    url: str,
    commit: str,
    expected_hash: bytes,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    def builder(_):
        dst = output()

        hasher = hasher_factory()

        # Clone next to the output (same filesystem, so the final move is a
        # rename); remove the clone if anything below fails.
        tempdir_path = Path(
            tempfile.mkdtemp(prefix=f"{filename}.clone-", dir=dst.parent)
        )
        try:
            repo = git.Repo.clone_from(url, tempdir_path)
            correct_commit = repo.create_head("correct_commit", commit)
            repo.head.reference = correct_commit
            assert not repo.head.is_detached
            # Reset the index and working tree to match the pointed-to commit.
            repo.head.reset(index=True, working_tree=True)
            repo.close()
            shutil.rmtree(tempdir_path / ".git")

            hash_directory_tree(hasher, tempdir_path)
            h = hasher.digest()
            if h != expected_hash:
                raise Exception(
                    f"Hash of git repo {filename} is wrong. "
                    f"Expected: {expected_hash.hex()}, actual: {h.hex()} "
                    f"(computed using {hasher})"
                )
        except BaseException:
            shutil.rmtree(tempdir_path, ignore_errors=True)
            raise

        shutil.move(tempdir_path, dst)

    return Derivation(filename, expected_hash, [], builder)


def LocalFile(filepath: Path) -> Derivation:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 16):
            hasher.update(chunk)
    h = hasher.digest()

    def builder(_):
        dst = output()

        shutil.copy(filepath, dst)

    return Derivation(filepath.name, h, [], builder)


def LocalSymlink(name: str, filepath: Path) -> Derivation:
    def builder(_):
        dst = output()

        dst.symlink_to(filepath)

    hasher = hashlib.sha256()
    hasher.update(str(filepath).encode("utf-8"))

    return Derivation(name, hasher.digest(), [], builder)


FileLike = Derivation | Expression[Path]


def Symlink(name: str, input: FileLike) -> Derivation:
    """A store entry that is a symlink to the realized ``input``."""

    @derivation(name)
    def builder(input: Path):
        dst = output()
        dst.symlink_to(input)

    return builder(input)


def Rename(name: str, input: Realizable) -> Derivation:
    @derivation(name)
    def builder(input: Path):
        dst = output()
        if input.is_file():
            shutil.copy(input, dst)
        elif input.is_dir():
            shutil.copytree(input, dst)
        else:
            raise Exception(f"don't know what to do with a file like {input}")

    return builder(input)


@expression()
def ChildFile(parent: Path, child: str) -> Path:
    return parent / child


@expression()
def Constant(x):
    return x


@expression()
def Gather(*nodes: Any) -> list[Any]:
    """The realized values of ``nodes`` as a list, in order. Its only job is
    to make several unrelated nodes the dependencies of one node, so that a
    single ``realize`` builds them all, concurrently when the executor allows."""
    return list(nodes)

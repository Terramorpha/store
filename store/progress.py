"""Progress reporting: one datagram socket per realization that every build
(in-thread, isolated child, or any script they launch) sends updates to.

Sending side -- ``progress()`` returns the handle for the current build. It
learns the socket name and the build id from the context variable the
scheduler binds around an in-process build, or else from the environment
(``STORE_PROGRESS_SOCKET``, ``STORE_BUILD_ID``) that the scheduler sets for
isolated children and that ``build_env()`` lets a builder pass on to the
scripts it launches. Outside any build the handle is a silent no-op.

Wire format -- one datagram per update, ``KEY=VALUE`` lines: ``ID`` (the
build label), ``DONE``, ``TOTAL`` (numbers), ``STATUS`` (free text, one
line). Sends are non-blocking and coalesced (at most one ``DONE``-only update
per 100 ms); a full queue or a dead receiver loses an update, never stalls a
build.

Receiving side -- :class:`ProgressServer`, owned by the scheduler: an
abstract-namespace Unix datagram socket (Linux; it disappears with the
process, nothing to clean up) read by one thread that parses messages and
hands ``(build_id, fields)`` to a callback.
"""

from __future__ import annotations

import errno
import os
import socket
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

ENV_SOCKET = "STORE_PROGRESS_SOCKET"
ENV_BUILD_ID = "STORE_BUILD_ID"

# (socket name, build id) for the build running on this thread, bound by the
# scheduler around in-process builders.
_BUILD: ContextVar[tuple[str, str] | None] = ContextVar("STORE_BUILD", default=None)

_COALESCE_S = 0.1


def _socket_address(name: str) -> bytes | str:
    """Abstract-namespace names are stored with a leading '@' in the
    environment (NUL cannot travel through an environment variable)."""
    return b"\0" + name[1:].encode() if name.startswith("@") else name


# --- state model (owned by the scheduler, rendered by reporters) ---------------


@dataclass
class BuildState:
    """What the scheduler knows about one node of the current realization."""

    label: str
    name: str
    pool: str
    status: str = (
        "pending"  # pending | waiting | running | done | cached | failed | blocked
    )
    started_at: float | None = None
    finished_at: float | None = None
    done: float | None = None
    total: float | None = None
    text: str = ""
    error: str = ""

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    @property
    def fraction(self) -> float | None:
        if self.done is None or not self.total:
            return None
        return max(0.0, min(1.0, self.done / self.total))


@dataclass
class Snapshot:
    """Immutable-by-convention view handed to reporters."""

    started_at: float
    builds: dict[str, BuildState] = field(default_factory=dict)

    def count(self, status: str) -> int:
        return sum(1 for b in self.builds.values() if b.status == status)


class Reporter:
    """What ``realize(..., reporter=...)`` expects: ``update`` is called with a
    fresh :class:`Snapshot` whenever something changed (throttled), ``close``
    once at the end. The default reporter does nothing."""

    def update(self, snapshot: Snapshot) -> None:  # noqa: B027 - intentional no-op
        pass

    def close(self, snapshot: Snapshot) -> None:  # noqa: B027
        pass


# --- client -----------------------------------------------------------------


class ProgressHandle:
    """The current build's progress channel. All methods are safe to call at
    any rate; updates are coalesced and never block."""

    def __init__(self, socket_name: str | None, build_id: str | None):
        self._addr = _socket_address(socket_name) if socket_name else None
        self._id = build_id
        self._sock: socket.socket | None = None
        self._done: float | None = None
        self._total: float | None = None
        self._text: str = ""
        self._last_sent = 0.0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._addr is not None and self._id is not None

    def total(self, total: float) -> None:
        """Set the denominator."""
        with self._lock:
            self._total = total
            self._send(force=True)

    def set(self, done: float) -> None:
        """Set the numerator."""
        with self._lock:
            self._done = done
            self._send(force=False)

    def status(self, text: str) -> None:
        """Set the one-line status text."""
        with self._lock:
            if text != self._text:
                self._text = text
                self._send(force=True)

    def _send(self, *, force: bool) -> None:
        if not self.active:
            return
        now = time.time()
        if not force and now - self._last_sent < _COALESCE_S:
            return
        self._last_sent = now
        lines = [f"ID={self._id}"]
        if self._done is not None:
            lines.append(f"DONE={self._done}")
        if self._total is not None:
            lines.append(f"TOTAL={self._total}")
        if self._text:
            lines.append("STATUS=" + self._text.replace("\n", " "))
        payload = ("\n".join(lines) + "\n").encode()
        try:
            if self._sock is None:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._sock.setblocking(False)
            self._sock.sendto(payload, self._addr)
        except OSError as e:
            # Full queue (EAGAIN) or nobody listening (ECONNREFUSED/ENOENT):
            # a progress update is never worth stalling or failing a build.
            if e.errno not in (
                errno.EAGAIN,
                errno.EWOULDBLOCK,
                errno.ECONNREFUSED,
                errno.ENOENT,
            ):
                raise


_NOOP = ProgressHandle(None, None)


def _current() -> tuple[str | None, str | None]:
    bound = _BUILD.get()
    if bound is not None:
        return bound
    return os.environ.get(ENV_SOCKET), os.environ.get(ENV_BUILD_ID)


def progress() -> ProgressHandle:
    """The progress handle of the build this code runs in (a no-op handle
    outside any build or when no realizer is listening)."""
    sock, bid = _current()
    if not sock or not bid:
        return _NOOP
    return ProgressHandle(sock, bid)


def build_env() -> dict[str, str]:
    """A copy of the environment carrying the current build's progress socket
    and id, for a builder to pass to the subprocesses it launches
    (``subprocess.run(..., env=build_env())``). Unchanged outside a build."""
    env = dict(os.environ)
    sock, bid = _current()
    if sock and bid:
        env[ENV_SOCKET] = sock
        env[ENV_BUILD_ID] = bid
    return env


# --- server -----------------------------------------------------------------


def parse_message(data: bytes) -> tuple[str | None, dict[str, str]]:
    fields: dict[str, str] = {}
    for line in data.decode(errors="replace").split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v
    return fields.pop("ID", None), fields


class ProgressServer:
    """One per realization. ``on_update(build_id, fields)`` is called on the
    reader thread for every message; fields are the raw strings."""

    def __init__(self, on_update):
        self.name = f"@store-progress-{os.getpid()}-{uuid4().hex[:12]}"
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(_socket_address(self.name))
        self._sock.settimeout(0.1)
        self._on_update = on_update
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="store-progress", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            bid, fields = parse_message(data)
            if bid is not None:
                try:
                    self._on_update(bid, fields)
                except Exception:  # noqa: BLE001 - a reporter bug must not kill the reader
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

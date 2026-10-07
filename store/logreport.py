"""Plain-file reporting for a realization, for readers that are not a
terminal: a person tailing a log, a script, a language model.

:class:`LogReporter` writes two files.

* ``log`` -- append-only, one line per state transition::

      14:36:42 done     gru_scratch_Restaurant_4_s0_3000k  pool=sim elapsed=2h15m
      14:36:50 failed   mentor_opd_OfficeSmall_3_s0  pool=sim elapsed=0:12  error=...

  plus a header when the realization starts and a summary line when it ends.
  ``grep done``, ``grep failed``, ``tail`` all do what one expects; nothing
  is ever rewritten.

* ``status`` -- the current state of the realization as one JSON object,
  rewritten atomically at most every ``period`` seconds and at the end:
  counts by status, elapsed, overall ETA, every running build with its
  elapsed time, fraction and ETA, the pending and failed builds, and the same
  one-line ``summary`` the dashboard shows. ``cat`` it to know where things
  are; it is a value, never a partial write.

Both paths are given explicitly; nothing is derived from the store or the
graph. Timestamps are local wall-clock, ``HH:MM:SS``, since the reader is on
the same machine as the build.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from store.progress import BuildState, Reporter, Snapshot, fmt_elapsed


def _clock(t: float | None = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _build_json(b: BuildState) -> dict:
    d = {
        "name": b.name,
        "label": b.label,
        "pool": b.pool,
        "slots": b.slots,
        "status": b.status,
        "elapsed_s": None if b.elapsed is None else round(b.elapsed, 1),
    }
    if b.fraction is not None:
        d["fraction"] = round(b.fraction, 4)
    if b.eta is not None:
        d["eta_s"] = round(b.eta, 1)
    if b.text:
        d["text"] = b.text
    if b.error:
        d["error"] = b.error
    return d


class LogReporter(Reporter):
    """See the module docstring. ``period`` bounds how often ``status`` is
    rewritten; the log is appended to as transitions happen."""

    def __init__(self, log: str | Path, status: str | Path, *, period: float = 30.0):
        self.log = Path(log)
        self.status = Path(status)
        self.period = period
        self._seen: dict[str, str] = {}
        self._last_status = 0.0
        self._opened = False

    # --- log -------------------------------------------------------------
    def _append(self, lines: list[str]) -> None:
        if not lines:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as fh:
            fh.write("".join(line + "\n" for line in lines))

    def _transitions(self, snap: Snapshot) -> list[str]:
        lines = []
        if not self._opened:
            self._opened = True
            lines.append(
                f"{_clock(snap.started_at)} start    {len(snap.builds)} nodes"
                + "".join(f"  {s} {n}" for s, n in snap.counts().items() if n)
            )
        for label, b in snap.builds.items():
            if self._seen.get(label) == b.status:
                continue
            self._seen[label] = b.status
            if b.status == "pending":
                continue
            line = f"{_clock()} {b.status:<8} {b.name}  pool={b.pool}"
            if b.slots != 1:
                line += f" x{b.slots}"
            if b.status in ("done", "failed") and b.elapsed is not None:
                line += f" elapsed={fmt_elapsed(b.elapsed)}"
            if b.status == "failed" and b.error:
                line += f"  error={b.error.strip().splitlines()[-1]}"
            lines.append(line)
        return lines

    # --- status ----------------------------------------------------------
    def _write_status(self, snap: Snapshot) -> None:
        builds = list(snap.builds.values())
        running = sorted(
            (b for b in builds if b.status in ("running", "waiting")),
            key=lambda b: b.started_at or 0,
        )
        eta = snap.eta()
        doc = {
            "time": _clock(),
            "started": _clock(snap.started_at),
            "elapsed_s": round(snap.elapsed, 1),
            "eta_s": None if eta is None else round(eta, 1),
            "summary": snap.summary(),
            "counts": snap.counts(),
            "running": [_build_json(b) for b in running],
            "pending": [b.name for b in builds if b.status == "pending"],
            "failed": [_build_json(b) for b in builds if b.status == "failed"],
            "blocked": [b.name for b in builds if b.status == "blocked"],
        }
        self.status.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status.with_name(self.status.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, self.status)
        self._last_status = time.time()

    # --- Reporter --------------------------------------------------------
    def update(self, snapshot: Snapshot) -> None:
        self._append(self._transitions(snapshot))
        if time.time() - self._last_status >= self.period:
            self._write_status(snapshot)

    def close(self, snapshot: Snapshot) -> None:
        lines = self._transitions(snapshot)
        lines.append(f"{_clock()} end      {snapshot.summary()}")
        self._append(lines)
        self._write_status(snapshot)

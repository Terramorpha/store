"""Terminal dashboard for a realization: a rich live table driven by the
scheduler's snapshots. Import this module only where a terminal UI is wanted;
the scheduler itself never imports rich."""

from __future__ import annotations

import threading

from rich.console import Console
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table

from store.progress import Reporter, Snapshot, fmt_elapsed


class RichReporter(Reporter):
    """Live table: pool occupancy, every running build with elapsed time, a
    bar when it reported a total, and its status text; counts underneath.
    Refreshes on its own thread a few times per second."""

    def __init__(
        self, *, refresh_per_second: float = 4.0, console: Console | None = None
    ):
        self._snapshot: Snapshot | None = None
        self._lock = threading.Lock()
        self._live = Live(
            console=console, refresh_per_second=refresh_per_second, transient=False
        )
        self._stop = threading.Event()
        self._period = 1.0 / refresh_per_second
        self._thread = threading.Thread(target=self._loop, name="store-ui", daemon=True)
        self._live.start()
        self._thread.start()

    def update(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def close(self, snapshot: Snapshot) -> None:
        self.update(snapshot)
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._live.update(self._render(snapshot))
        self._live.stop()

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            with self._lock:
                snap = self._snapshot
            if snap is not None:
                self._live.update(self._render(snap))

    def _render(self, snap: Snapshot):
        running = [
            b for b in snap.builds.values() if b.status in ("running", "waiting")
        ]
        running.sort(key=lambda b: b.started_at or 0)
        table = Table(expand=True, show_edge=False, pad_edge=False)
        table.add_column("build", ratio=3, no_wrap=True)
        table.add_column("pool", width=10)
        table.add_column("elapsed", width=8, justify="right")
        table.add_column("eta", width=8, justify="right")
        table.add_column("progress", ratio=2)
        table.add_column("status", ratio=3, no_wrap=True)
        for b in running:
            if b.fraction is not None:
                bar = ProgressBar(total=1.0, completed=b.fraction, width=None)
            else:
                bar = "waiting for lock" if b.status == "waiting" else ""
            eta = fmt_elapsed(b.eta) if b.eta is not None else ""
            pool = b.pool if b.slots == 1 else f"{b.pool} x{b.slots}"
            table.add_row(b.name, pool, fmt_elapsed(b.elapsed), eta, bar, b.text)
        table.caption = snap.summary()
        return table

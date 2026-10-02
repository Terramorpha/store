"""A synthetic build graph to look at the dashboard.

    uv run python examples/demo_ui.py [store_dir]

Three kinds of leaves (fast, slow, isolated), two pools, a join over all of
them, and a final node. Leaves report progress; the slow ones also report a
status line. Run it twice: the second run is all cache hits.
"""

import os
import random
import sys
import tempfile
import time
from pathlib import Path

from store import Gather, LocalExecutor, derivation, output, progress, realize
from store.ui import RichReporter


@derivation(lambda i: f"fast{i}.txt", pool="light")
def fast(i: int) -> None:
    p = progress()
    p.total(10)
    for k in range(10):
        time.sleep(0.15 + random.random() * 0.1)
        p.set(k + 1)
    output().write_text(f"fast {i}")


@derivation(lambda i: f"slow{i}.txt", pool="heavy")
def slow(i: int) -> None:
    p = progress()
    n = 40
    p.total(n)
    p.status("warming up")
    for k in range(n):
        time.sleep(0.1 + random.random() * 0.05)
        p.set(k + 1)
        p.status(f"epoch {k + 1}/{n}  loss {1.0 / (k + 1):.3f}")
    output().write_text(f"slow {i}")


@derivation(lambda i: f"isolated{i}.txt", pool="heavy", isolate=True)
def isolated(i: int) -> None:
    p = progress()
    p.total(20)
    p.status(f"child pid {os.getpid()}")
    for k in range(20):
        time.sleep(0.15)
        p.set(k + 1)
    output().write_text(f"isolated {i}")


@derivation(lambda parts: "join.txt")
def join(parts: list[Path]) -> None:
    time.sleep(1.0)
    output().write_text("\n".join(p.read_text() for p in parts))


@derivation(lambda j: "report.txt")
def report(j: Path) -> None:
    p = progress()
    p.status("writing the report")
    time.sleep(1.5)
    output().write_text(j.read_text().upper())


def main() -> None:
    if len(sys.argv) > 1:
        store_dir = Path(sys.argv[1])
    else:
        store_dir = Path(tempfile.mkdtemp(prefix="store-demo-"))
    leaves = (
        [fast(i) for i in range(12)]
        + [slow(i) for i in range(4)]
        + [isolated(i) for i in range(3)]
    )
    root = report(join(Gather(*leaves)))
    ex = LocalExecutor({"light": 4, "heavy": 3, "default": 2})
    out = realize(store_dir, root, executor=ex, reporter=RichReporter())
    print("result:", out)


if __name__ == "__main__":
    main()

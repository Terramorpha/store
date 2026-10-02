import tempfile
import time
from pathlib import Path

from store import Gather, LocalExecutor, derivation, output, realize
from store.progress import progress
from store.ui import RichReporter


@derivation(lambda n: f"waiter_{n}")
def thing(n):

    p = progress()
    p.total(n)

    for i in range(n):
        p.set(i + 1)
        time.sleep(1)

    output().write_text("done")


@derivation("grouper")
def dependent(*things):
    print(things)
    output().write_text("done")


things = [thing(i) for i in range(100)]


ex = LocalExecutor({"default": 4})


tmpdir = tempfile.mkdtemp()

realize(
    Path(tmpdir).resolve(True),
    Gather(*things),
    executor=ex,
    reporter=RichReporter(),
)

# store

A tiny content-addressed build store, Nix in miniature, extracted from
[Building2Building](https://github.com/vtaboga/Building2Building)'s data
pipeline so that other projects can use it for the artifacts their experiments
consume and produce.

A **derivation** is a named build step: a hash, a list of dependencies and a
builder. `realize(store_path, derivation)` builds it into
`<store_path>/<hash>-<name>` unless that path already exists, building into a
temporary path first and renaming atomically so that concurrent processes can
never see a half-built output or clobber each other's. An **expression** is
the same idea without caching: it is evaluated every time.

The decorators turn an ordinary function into a graph node. Arguments that are
themselves derivations or expressions become dependencies (realized first and
passed in as paths / values); every other argument is folded into the hash,
together with the function's own source text, so editing a step invalidates
its outputs.

```python
from pathlib import Path
from store import DownloadFile, LocalExecutor, derivation, output, realize

archive = DownloadFile("data.tar.gz", "https://example.org/data.tar.gz", hash=None)


@derivation("summary.json")
def summarize(archive: Path, threshold: float) -> None:
    out = output()  # where to write; bound by realize()
    ...
    out.write_text("...")


path = realize(
    Path("~/.cache/mystore").expanduser(),
    summarize(archive, threshold=0.5),
    executor=LocalExecutor(),   # serial; always given explicitly
)
```

Hashes are **input-addressed**: they name the recipe (function, arguments,
dependency hashes), not the bytes the recipe produced, so non-deterministic
builders (training runs, simulations) are fine.

Included concrete derivations: `DownloadFile`, `GitClone`, `LocalFile`,
`LocalSymlink`, `Symlink`, `Rename`, `ExtractTarball`, `ExtractZip`,
`ExtractFromZip`, and the expressions `ChildFile`, `Constant`, `Gather`.

## Parallel realization

`realize` takes one node and an executor (always explicit; `LocalExecutor()`
is the serial one). The DAG under the node is scheduled
in dependency order; its independent parts run concurrently within the slot
budget of their **pool**. Parallelism is therefore automatic: a derivation
whose inputs do not depend on each other gets them built side by side. To
build several unrelated nodes together, make them the inputs of one node,
e.g. `Gather(a, b, c)` (an expression returning their values as a list):

```python
from store import LocalExecutor, derivation, realize


@derivation("checkpoint", pool="train", isolate=True)
def train(config: dict, seed: int) -> None: ...


@derivation("eval.csv", pool="eval")
def evaluate(checkpoint: Path) -> None: ...


ex = LocalExecutor({"train": 3, "eval": 1}, default_pool_size=8)
evals = realize(store, Gather(*[evaluate(train(cfg, s)) for s in range(3)]), executor=ex)
```

* `pool`: the executor pool whose slot the build occupies (unlisted pools get
  `default_pool_size`). `LocalExecutor()` with no arguments is serial.
* `isolate=True`: the builder runs in a fresh interpreter (the closure and its
  realized inputs travel by cloudpickle), for crash isolation and for
  libraries that must not share process state.
* Failures: a failed derivation blocks its dependents and nothing else; when
  everything runnable has run, `RealizeError` lists the failed and blocked
  nodes. `fail_fast=True` raises at the first failure without waiting for the
  builds still running.
* Build locks: a `<output>.lock` file with a heartbeat marks an output being
  built, so another realizer (thread, process, or a driver restarted after a
  crash) waits for it instead of building it again; a lock whose heartbeat is
  older than `lock_stale_after` is taken over.

## Development

```
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest
```

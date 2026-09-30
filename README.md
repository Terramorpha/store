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
from store import OUTPUT, DownloadFile, derivation, realize

archive = DownloadFile("data.tar.gz", "https://example.org/data.tar.gz", hash=None)

@derivation("summary.json")
def summarize(archive: Path, threshold: float) -> None:
    out = OUTPUT.get()          # where to write; set by realize()
    ...
    out.write_text("...")

path = realize(Path("~/.cache/mystore").expanduser(), summarize(archive, threshold=0.5))
```

Hashes are **input-addressed**: they name the recipe (function, arguments,
dependency hashes), not the bytes the recipe produced, so non-deterministic
builders (training runs, simulations) are fine.

Included concrete derivations: `DownloadFile`, `GitClone`, `LocalFile`,
`LocalSymlink`, `Symlink`, `Rename`, `ExtractTarball`, `ExtractZip`,
`ExtractFromZip`, and the expressions `ChildFile`, `Constant`.

## Development

```
uv sync
uv run pytest
```

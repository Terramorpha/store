"""Child-interpreter entry point for ``isolate=True`` derivations.

``python -m store._isolated <payload.pkl>``: the payload is a cloudpickled
``(builder, realized_deps, output_path)``; we bind OUTPUT and call the builder.
Any exception propagates as a non-zero exit with the traceback on stderr.
"""

import sys
from pathlib import Path

import cloudpickle

from store.core import OUTPUT


def main() -> None:
    with open(sys.argv[1], "rb") as f:
        builder, realized_deps, output_path = cloudpickle.load(f)
    OUTPUT.set(Path(output_path))
    builder(realized_deps)


if __name__ == "__main__":
    main()

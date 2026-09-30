"""The graph layer: nodes (derivations, expressions), the decorators that
make them from Python functions, hashing, and output() for builders.
Everything public is re-exported from ``store``.
"""

import hashlib
import inspect
import pickle
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Generic, TypeVar

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

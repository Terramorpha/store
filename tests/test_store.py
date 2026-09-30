"""Behavioural tests for the store: caching, hashing, atomicity and the stock
derivations. Network and git are exercised through a fake ``requests`` and a
local repository, never the real network."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path

import git
import pytest

from store import (
    output,
    ChildFile,
    Constant,
    Derivation,
    DownloadFile,
    Expression,
    ExtractFromZip,
    ExtractTarball,
    ExtractZip,
    GitClone,
    LocalFile,
    LocalSymlink,
    Rename,
    Symlink,
    derivation,
    expression,
    hash_directory_tree,
    realize,
)
import store.core as core


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "store"


# --- derivations ------------------------------------------------------------


def test_derivation_is_built_once_and_named_by_hash(store_path: Path):
    calls: list[int] = []

    @derivation("greeting.txt")
    def greet(who: str) -> None:
        calls.append(1)
        output().write_text(f"hello {who}")

    d = greet("world")
    assert isinstance(d, Derivation)
    out = realize(store_path, d)
    assert out.parent == store_path
    assert out.name == d.hash.hex() + "-greeting.txt"
    assert out.read_text() == "hello world"
    assert realize(store_path, greet("world")) == out
    assert calls == [1]


def test_hash_depends_on_arguments_and_function_source(store_path: Path):
    @derivation("a.txt")
    def f(x: int) -> None:
        output().write_text(str(x))

    @derivation("a.txt")
    def g(x: int) -> None:
        output().write_text(str(x + 0))  # different source text

    assert f(1).hash != f(2).hash
    assert f(1).hash != g(1).hash
    assert f(1).hash == f(1).hash


def test_realizable_arguments_become_dependencies_and_feed_the_hash(store_path: Path):
    @derivation("base.txt")
    def base(n: int) -> None:
        output().write_text("x" * n)

    @derivation("len.txt")
    def length(base_path: Path) -> None:
        output().write_text(str(len(base_path.read_text())))

    d = length(base(3))
    assert [dep.hash for dep in d.dependencies] == [base(3).hash]
    assert length(base(3)).hash != length(base(4)).hash
    assert realize(store_path, d).read_text() == "3"
    assert (store_path / (base(3).hash.hex() + "-base.txt")).exists()


def test_derivation_name_can_be_computed_from_arguments(store_path: Path):
    @derivation(lambda tag: f"{tag}.txt")
    def f(tag: str) -> None:
        output().write_text(tag)

    assert f("abc").name == "abc.txt"
    assert realize(store_path, f("abc")).name.endswith("-abc.txt")


def test_derivation_name_cannot_contain_a_slash():
    with pytest.raises(AssertionError):
        Derivation("a/b", b"\x00", [], lambda deps: None)


def test_builder_that_writes_nothing_fails_loudly(store_path: Path):
    @derivation("nothing")
    def f() -> None:
        pass

    with pytest.raises(Exception, match="did not produce an output"):
        realize(store_path, f())
    assert not list(store_path.iterdir())  # nothing half-built left behind as final


def test_partial_temp_output_is_not_mistaken_for_a_build(store_path: Path):
    @derivation("out.txt")
    def f() -> None:
        output().write_text("ok")

    d = f()
    store_path.mkdir()
    stale = store_path / f"{d.hash.hex()}-out.txt.tmp-1-deadbeef"
    stale.write_text("half")
    out = realize(store_path, d)
    assert out.read_text() == "ok"
    assert stale.exists()  # left for a gc to clean; never confused with the result


def test_builder_output_can_be_a_directory(store_path: Path):
    @derivation("tree")
    def f() -> None:
        out = output()
        out.mkdir()
        (out / "a").write_text("a")

    out = realize(store_path, f())
    assert out.is_dir() and (out / "a").read_text() == "a"


def test_output_contextvar_is_reset_after_realization(store_path: Path):
    @derivation("x")
    def f() -> None:
        output().write_text("x")

    realize(store_path, f())
    with pytest.raises(RuntimeError, match="outside of a builder"):
        output()


# --- expressions ------------------------------------------------------------


def test_expression_returns_a_value_and_is_not_cached(store_path: Path):
    calls: list[int] = []

    @expression()
    def double(x: int) -> int:
        calls.append(1)
        return 2 * x

    e = double(21)
    assert isinstance(e, Expression)
    assert realize(store_path, e) == 42
    assert realize(store_path, e) == 42
    assert calls == [1, 1]


def test_expression_over_a_derivation(store_path: Path):
    @derivation("n.txt")
    def n() -> None:
        output().write_text("7")

    @expression()
    def read(p: Path) -> int:
        return int(p.read_text())

    assert realize(store_path, read(n())) == 7


def test_child_file_and_constant(store_path: Path):
    @derivation("dir")
    def d() -> None:
        out = output()
        out.mkdir()
        (out / "inner.txt").write_text("inner")

    assert realize(store_path, ChildFile(d(), "inner.txt")).read_text() == "inner"
    assert realize(store_path, Constant({"k": 1})) == {"k": 1}


# --- stock derivations ------------------------------------------------------


def test_local_file_copies_and_hashes_content(tmp_path: Path, store_path: Path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload" * 1000)
    d = LocalFile(src)
    assert d.hash == hashlib.sha256(src.read_bytes()).digest()
    out = realize(store_path, d)
    assert out.read_bytes() == src.read_bytes()
    assert out.name.endswith("-src.bin")


def test_local_symlink_points_at_the_path(tmp_path: Path, store_path: Path):
    target = tmp_path / "target"
    target.write_text("t")
    out = realize(store_path, LocalSymlink("link", target))
    assert out.is_symlink() and out.resolve() == target.resolve()


def test_symlink_points_at_its_input_not_itself(tmp_path: Path, store_path: Path):
    src = tmp_path / "f.txt"
    src.write_text("f")
    out = realize(store_path, Symlink("f-link", LocalFile(src)))
    assert out.is_symlink()
    assert out.resolve() != out and out.read_text() == "f"


def test_rename_copies_files_and_directories(tmp_path: Path, store_path: Path):
    src = tmp_path / "orig.txt"
    src.write_text("o")
    assert realize(store_path, Rename("renamed.txt", LocalFile(src))).read_text() == "o"

    @derivation("d")
    def d() -> None:
        out = output()
        out.mkdir()
        (out / "x").write_text("x")

    out = realize(store_path, Rename("copy", d()))
    assert out.name.endswith("-copy") and (out / "x").read_text() == "x"


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_extract_zip_and_extract_from_zip(tmp_path: Path, store_path: Path):
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(_zip_bytes({"a.txt": b"A", "sub/b.txt": b"B"}))
    z = LocalFile(archive)
    out = realize(store_path, ExtractZip(z))
    assert out.name.endswith("-bundle")
    assert (out / "a.txt").read_bytes() == b"A" and (
        out / "sub" / "b.txt"
    ).read_bytes() == b"B"
    one = realize(store_path, ExtractFromZip(z, "sub/b.txt"))
    assert one.name.endswith("-b.txt") and one.read_bytes() == b"B"
    assert ExtractFromZip(z, "a.txt").hash != ExtractFromZip(z, "sub/b.txt").hash


def test_extract_tarball(tmp_path: Path, store_path: Path):
    src = tmp_path / "content"
    src.mkdir()
    (src / "c.txt").write_text("C")
    archive = tmp_path / "content.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src / "c.txt", arcname="c.txt")
    out = realize(store_path, ExtractTarball(LocalFile(archive)))
    assert out.name.endswith("-content") and (out / "c.txt").read_text() == "C"


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"content-length": str(len(data))}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, block_size: int):
        for i in range(0, len(self._data), block_size):
            yield self._data[i : i + block_size]


def test_download_file_verifies_hash(monkeypatch, store_path: Path):
    data = b"downloaded bytes" * 10
    monkeypatch.setattr(
        core.requests, "get", lambda url, stream, verify: _FakeResponse(data)
    )
    good = hashlib.sha256(data).digest()
    out = realize(
        store_path, DownloadFile("blob.bin", "https://example.invalid/blob", good)
    )
    assert out.read_bytes() == data
    with pytest.raises(Exception, match="Hash of download"):
        realize(
            store_path,
            DownloadFile("blob2.bin", "https://example.invalid/blob2", b"\x00" * 32),
        )
    assert not any(p.name.endswith("-blob2.bin") for p in store_path.iterdir())


def test_download_file_without_hash_is_addressed_by_url(monkeypatch, store_path: Path):
    monkeypatch.setattr(
        core.requests, "get", lambda url, stream, verify: _FakeResponse(b"x")
    )
    d = DownloadFile("f", "https://example.invalid/f", None)
    assert d.hash == hashlib.sha256(b"https://example.invalid/f").digest()
    assert realize(store_path, d).read_bytes() == b"x"


def test_git_clone_checks_out_commit_and_verifies_tree_hash(
    tmp_path: Path, store_path: Path
):
    src = tmp_path / "repo"
    repo = git.Repo.init(src)
    (src / "README").write_text("v1")
    repo.index.add(["README"])
    c1 = repo.index.commit("v1")
    (src / "README").write_text("v2")
    repo.index.add(["README"])
    repo.index.commit("v2")

    expected_tree = tmp_path / "expected"
    expected_tree.mkdir()
    (expected_tree / "README").write_text("v1")
    hasher = hashlib.sha256()
    hash_directory_tree(hasher, expected_tree)
    expected = hasher.digest()

    out = realize(store_path, GitClone("repo", str(src), c1.hexsha, expected))
    assert (out / "README").read_text() == "v1"
    assert not (out / ".git").exists()

    with pytest.raises(Exception, match="Hash of git repo"):
        realize(store_path, GitClone("repo-bad", str(src), c1.hexsha, b"\x01" * 32))
    assert not any(
        "clone-" in p.name for p in store_path.iterdir()
    )  # temp clone removed


def test_hash_directory_tree_is_order_independent_and_path_sensitive(tmp_path: Path):
    a = tmp_path / "a"
    (a / "sub").mkdir(parents=True)
    (a / "sub" / "f").write_text("1")
    (a / "g").write_text("2")
    b = tmp_path / "b"
    (b / "sub").mkdir(parents=True)
    (b / "g").write_text("2")
    (b / "sub" / "f").write_text("1")

    def h(d: Path) -> bytes:
        hs = hashlib.sha256()
        hash_directory_tree(hs, d)
        return hs.digest()

    assert h(a) == h(b)
    (b / "sub" / "f").rename(b / "sub" / "f2")
    assert h(a) != h(b)

"""Stock derivations and expressions: downloads, git clones, local files
and symlinks, archives, and the glue expressions ChildFile / Constant / Gather.
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import git
import requests
from rich.progress import DownloadColumn, Progress, TransferSpeedColumn

from store.graph import (
    Derivation,
    Expression,
    Realizable,
    derivation,
    expression,
    output,
)


def DownloadFile(
    filename: str,
    url: str,
    hash: bytes | None,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    if hash is not None:
        derivation_hash = hash
    else:
        h = hashlib.sha256()
        h.update(url.encode("utf-8"))
        derivation_hash = h.digest()

    def builder(_):
        out = output()

        # Get progress object and create a task

        with Progress(
            *Progress.get_default_columns(),
            DownloadColumn(),
            TransferSpeedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading {filename}", total=None)

            hasher = hasher_factory()
            # Verify TLS certificates by default.
            #
            # If your environment requires a custom CA bundle, set one of:
            # - B2B_CA_BUNDLE=/path/to/ca-bundle.pem
            # - REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
            # - SSL_CERT_FILE=/path/to/ca-bundle.pem
            #
            # To explicitly opt-out (NOT recommended), set:
            # - B2B_INSECURE_SSL=1
            verify: bool | str = True
            if os.environ.get("B2B_INSECURE_SSL", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                verify = False
            else:
                ca_bundle = (
                    os.environ.get("B2B_CA_BUNDLE")
                    or os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("SSL_CERT_FILE")
                )
                if ca_bundle:
                    verify = ca_bundle

            response = requests.get(url, stream=True, verify=verify)
            response.raise_for_status()

            # Update task with actual file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                progress.update(task, total=total_size)

            block_size = 1 << 16

            with tempfile.NamedTemporaryFile("wb", delete=False) as outfile:
                for data in response.iter_content(block_size):
                    hasher.update(data)
                    outfile.write(data)
                    progress.update(task, advance=len(data))

            outfile.close()
            h = hasher.digest()
            if hash is not None and hash != h:
                raise Exception(
                    f"Hash of download {filename} is wrong. "
                    f"Expected: {hash.hex()}, actual: {h.hex()} "
                    f"(computed using {hasher})"
                )
            shutil.move(outfile.name, out)

    return Derivation(filename, derivation_hash, [], builder)


def ExtractTarball(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".tar.gz"))
    def inner(input: Path):
        dst = output()

        with tarfile.open(input, "r:gz") as tar:
            tar.extractall(path=dst)

    return inner(input_der)


def ExtractZip(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".zip"))
    def inner(input: Path):
        dst = output()

        dst.mkdir()

        with zipfile.ZipFile(input, "r") as zip_ref:
            file_list = zip_ref.infolist()

            for file_info in file_list:
                zip_ref.extract(file_info, dst)

    return inner(input_der)


def ExtractFromZip(zip_file: Realizable, filename: str) -> Derivation:
    @derivation(Path(filename).name)
    def inner(input: Path, name: str):
        dst = output()
        with zipfile.ZipFile(input, "r") as zip_ref:
            with zip_ref.open(name) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)

    return inner(zip_file, filename)


def hash_directory_tree(hasher, dir: Path):
    # Get all files and sort them for deterministic ordering
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()
        for file in files:
            file_paths.append(os.path.join(root, file))

    # Sort all file paths to ensure deterministic order
    file_paths.sort()

    # Hash each file's content
    for file_path in file_paths:
        # Include the relative path in the hash for structure integrity
        rel_path = os.path.relpath(file_path, dir)
        hasher.update(rel_path.encode("utf-8"))

        # Hash the file content
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)


def GitClone(
    filename: str,
    url: str,
    commit: str,
    expected_hash: bytes,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    def builder(_):
        dst = output()

        hasher = hasher_factory()

        # Clone next to the output (same filesystem, so the final move is a
        # rename); remove the clone if anything below fails.
        tempdir_path = Path(
            tempfile.mkdtemp(prefix=f"{filename}.clone-", dir=dst.parent)
        )
        try:
            repo = git.Repo.clone_from(url, tempdir_path)
            correct_commit = repo.create_head("correct_commit", commit)
            repo.head.reference = correct_commit
            assert not repo.head.is_detached
            # Reset the index and working tree to match the pointed-to commit.
            repo.head.reset(index=True, working_tree=True)
            repo.close()
            shutil.rmtree(tempdir_path / ".git")

            hash_directory_tree(hasher, tempdir_path)
            h = hasher.digest()
            if h != expected_hash:
                raise Exception(
                    f"Hash of git repo {filename} is wrong. "
                    f"Expected: {expected_hash.hex()}, actual: {h.hex()} "
                    f"(computed using {hasher})"
                )
        except BaseException:
            shutil.rmtree(tempdir_path, ignore_errors=True)
            raise

        shutil.move(tempdir_path, dst)

    return Derivation(filename, expected_hash, [], builder)


def LocalFile(filepath: Path) -> Derivation:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 16):
            hasher.update(chunk)
    h = hasher.digest()

    def builder(_):
        dst = output()

        shutil.copy(filepath, dst)

    return Derivation(filepath.name, h, [], builder)


def LocalSymlink(name: str, filepath: Path) -> Derivation:
    def builder(_):
        dst = output()

        dst.symlink_to(filepath)

    hasher = hashlib.sha256()
    hasher.update(str(filepath).encode("utf-8"))

    return Derivation(name, hasher.digest(), [], builder)


FileLike = Derivation | Expression[Path]


def Symlink(name: str, input: FileLike) -> Derivation:
    """A store entry that is a symlink to the realized ``input``."""

    @derivation(name)
    def builder(input: Path):
        dst = output()
        dst.symlink_to(input)

    return builder(input)


def Rename(name: str, input: Realizable) -> Derivation:
    @derivation(name)
    def builder(input: Path):
        dst = output()
        if input.is_file():
            shutil.copy(input, dst)
        elif input.is_dir():
            shutil.copytree(input, dst)
        else:
            raise Exception(f"don't know what to do with a file like {input}")

    return builder(input)


@expression()
def ChildFile(parent: Path, child: str) -> Path:
    return parent / child


@expression()
def Constant(x):
    return x


@expression()
def Gather(*nodes: Any) -> list[Any]:
    """The realized values of ``nodes`` as a list, in order. Its only job is
    to make several unrelated nodes the dependencies of one node, so that a
    single ``realize`` builds them all, concurrently when the executor allows."""
    return list(nodes)

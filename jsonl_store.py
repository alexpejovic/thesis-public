"""Append-only JSONL with cross-node locking.

Why locking is needed here specifically
---------------------------------------
``encoding/data`` is a symlink shared between this repo's git worktrees, and the
result and ledger files are written by SLURM array tasks running concurrently on
different nodes. So the writers are separate processes on separate machines
appending to one file.

The older pattern in this repo (one process, ``open(..., "a")`` plus ``flush()``
from a ``ThreadPoolExecutor``) is safe only because it is a single process. It
must not be copied for an array job.

This Lustre mount is mounted with ``flock`` (not ``localflock``), so
``fcntl.flock`` is honoured across nodes. ``probe_flock`` checks that at startup
rather than assuming it, and if locking is genuinely unavailable the writers fall
back to one file per row under a ``.d`` directory, created via ``os.link`` --
which is atomic on Lustre regardless of mount options. ``load_rows`` reads both
layouts, so the fallback needs no separate merge step.

Design notes, each load-bearing:

* ``json.dumps`` happens *outside* the lock, so the critical section is one
  ``write`` plus one ``fsync``.
* the lock is a **sidecar** ``.lock`` file opened ``"a"`` (never truncated), so
  the resume-read and the append share one lock without the reader needing to
  open the data file for writing, and rewriting the data file by rename cannot
  orphan the lock.
* the resume-read takes ``LOCK_EX``, not ``LOCK_SH``: a shared lock would still
  permit observing a concurrent appender's partially written line.
* the reader also *tolerates* a torn final line rather than raising. That is the
  real failure mode of a ``scancel``'d task, and one dead task must not make the
  whole file unreadable.
* ``os.fsync`` because a single row can represent hours of GPU time.
"""

import fcntl
import glob
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def shard_dir(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".d")


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path``'s sidecar lock file."""
    lock = _lock_path(Path(path))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as handle:  # "a" never truncates
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def probe_flock(path: Path) -> bool:
    """True if this filesystem honours flock. Cheap; call once at startup.

    Only a genuine *locking* failure counts as False. A path that cannot be
    created at all is a caller error and must propagate: swallowing it here
    silently routes writes to the shard fallback, which then fails on the same
    unusable directory with a far more confusing traceback.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create the directory for {path}: {exc}") from exc
    try:
        with locked(path):
            return True
    except OSError:
        return False


def append_row(path: str | Path, row: dict, *, use_lock: bool | None = None) -> None:
    """Append one JSON object as a line. Safe against concurrent writers."""
    path = Path(path)
    line = json.dumps(row) + "\n"  # serialize BEFORE taking the lock
    if use_lock is None:
        use_lock = probe_flock(path)

    if not use_lock:
        _append_shard(path, line)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        with open(path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def _append_shard(path: Path, line: str) -> None:
    """Fallback: one file per row, linked into place so it cannot collide."""
    d = shard_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(line.encode()).hexdigest()[:20]
    target = d / f"{digest}.json"
    if target.exists():
        return
    tmp = d / f".{digest}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    try:
        # O_EXCL semantics via link(): fails if the target already exists.
        os.link(tmp, target)
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)


def load_rows(path: str | Path) -> list[dict]:
    """Every row, from the main file and any shard directory.

    A truncated or malformed line is skipped, not raised on: a task killed at
    the wall clock can leave a partial final line, and that must not make the
    file unreadable.
    """
    path = Path(path)
    rows: list[dict] = []

    if path.exists():
        with locked(path):
            text = path.read_text()
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line from a killed task

    for shard in sorted(glob.glob(str(shard_dir(path) / "*.json"))):
        try:
            rows.append(json.loads(Path(shard).read_text()))
        except (json.JSONDecodeError, OSError):
            continue

    return rows


def row_key(row: dict, fields: Iterable[str]) -> tuple:
    """A hashable resume key from the named fields.

    Lists become tuples so a key survives the JSON round trip -- without this,
    ``["Na_gNa"]`` read back from disk would never equal the in-memory key.
    """
    out = []
    for f in fields:
        v = row.get(f)
        out.append(tuple(v) if isinstance(v, list) else v)
    return tuple(out)


def existing_keys(path: str | Path, fields: Iterable[str]) -> set[tuple]:
    fields = list(fields)
    return {row_key(r, fields) for r in load_rows(path)}


def collect(path: str | Path, fields: Iterable[str]) -> int:
    """De-duplicate by key (last wins) and rewrite atomically.

    Folds any shard files into the main file, drops torn lines, and replaces the
    file by rename so a reader never sees a partial rewrite. Returns the number
    of rows kept.
    """
    path = Path(path)
    fields = list(fields)
    rows = load_rows(path)
    deduped: dict[tuple, dict] = {}
    for r in rows:
        deduped[row_key(r, fields)] = r

    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            for r in deduped.values():
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic within a directory

    d = shard_dir(path)
    if d.is_dir():
        for shard in glob.glob(str(d / "*.json")):
            Path(shard).unlink(missing_ok=True)

    return len(deduped)

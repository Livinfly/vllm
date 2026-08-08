# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a portable experiment-only SQLite trace from an nsys export."""

import argparse
import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

MARKER_PREFIX = "VLLM_PCP_DCP_MLA|"
SENSITIVE_NAME_PARTS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sensitive_environment() -> dict[str, bytes]:
    return {
        name: value.encode()
        for name, value in os.environ.items()
        if len(value) >= 8
        and any(part in name.upper() for part in SENSITIVE_NAME_PARTS)
    }


def verify_no_sensitive_environment(path: Path) -> None:
    """Fail if a file contains a current credential-like environment value."""
    sensitive = _sensitive_environment()
    leaked: set[str] = set()
    overlap = max((len(value) for value in sensitive.values()), default=1) - 1
    tail = b""
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            content = tail + chunk
            leaked.update(name for name, value in sensitive.items() if value in content)
            tail = content[-overlap:] if overlap else b""
    if leaked:
        names = ", ".join(sorted(leaked))
        raise RuntimeError(f"{path} contains sensitive environment values: {names}")


def _string_map(connection: sqlite3.Connection) -> dict[int, str]:
    return dict(connection.execute("SELECT id, value FROM StringIds"))


def _create_portable_database(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    source_sqlite_sha256: str,
    source_report_sha256: str | None,
) -> None:
    destination.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, globalTid INTEGER, text TEXT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
            streamId INTEGER, correlationId INTEGER, globalPid INTEGER,
            demangledName INTEGER, shortName INTEGER
        );
        CREATE TABLE EXPERIMENT_TRACE_METADATA (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        """
    )

    strings = _string_map(source)
    markers = []
    for row in source.execute(
        "SELECT rowid, start, end, globalTid, text, textId FROM NVTX_EVENTS"
    ):
        rowid, start, end, global_tid, text, text_id = row
        message = text or strings.get(text_id)
        if message and message.startswith(MARKER_PREFIX):
            markers.append((rowid, start, end, global_tid, message))
    if not markers:
        raise RuntimeError(f"No {MARKER_PREFIX} markers were found")
    destination.executemany(
        "INSERT INTO NVTX_EVENTS(rowid, start, end, globalTid, text) "
        "VALUES (?, ?, ?, ?, ?)",
        markers,
    )

    runtime_rows = list(
        source.execute(
            "SELECT rowid, start, end, globalTid, correlationId, nameId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    kernel_rows = list(
        source.execute(
            "SELECT rowid, start, end, deviceId, contextId, streamId, "
            "correlationId, globalPid, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    destination.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME("
        "rowid, start, end, globalTid, correlationId, nameId) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        runtime_rows,
    )
    destination.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL("
        "rowid, start, end, deviceId, contextId, streamId, correlationId, "
        "globalPid, demangledName, shortName) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        kernel_rows,
    )

    referenced_string_ids = {row[-1] for row in runtime_rows if row[-1] is not None}
    for row in kernel_rows:
        referenced_string_ids.update(value for value in row[-2:] if value is not None)
    missing = referenced_string_ids.difference(strings)
    if missing:
        raise RuntimeError(f"Missing referenced StringIds: {sorted(missing)}")
    destination.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (string_id, strings[string_id])
            for string_id in sorted(referenced_string_ids)
        ],
    )

    metadata = [
        ("format", "vllm-pcp-dcp-mla-portable-sqlite-v1"),
        ("source_sqlite_sha256", source_sqlite_sha256),
        ("experiment_marker_count", str(len(markers))),
        ("runtime_row_count", str(len(runtime_rows))),
        ("kernel_row_count", str(len(kernel_rows))),
    ]
    if source_report_sha256 is not None:
        metadata.append(("source_report_sha256", source_report_sha256))
    destination.executemany(
        "INSERT INTO EXPERIMENT_TRACE_METADATA(key, value) VALUES (?, ?)", metadata
    )
    destination.commit()
    result = destination.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError(f"Portable SQLite integrity check failed: {result}")


def sanitize(
    input_path: Path,
    output_path: Path,
    source_report: Path | None,
    force: bool,
) -> None:
    """Write a portable trace database without nsys environment metadata."""
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{input_path}?mode=ro", uri=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        destination = sqlite3.connect(temporary_path)
        try:
            _create_portable_database(
                source,
                destination,
                _sha256(input_path),
                _sha256(source_report) if source_report is not None else None,
            )
        finally:
            source.close()
            destination.close()
        verify_no_sensitive_environment(temporary_path)
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source.sqlite"
        output_path = root / "portable.sqlite"
        source = sqlite3.connect(source_path)
        source.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER,
                text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
                streamId INTEGER, correlationId INTEGER, globalPid INTEGER,
                demangledName INTEGER, shortName INTEGER
            );
            """
        )
        source.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(1, "cudaLaunchKernel"), (2, "kernel"), (3, "discard-me")],
        )
        source.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
            (1, 2, 3, f"{MARKER_PREFIX}category=scope|iteration=i|rank=0", None),
        )
        source.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            (1, 2, 3, 4, 1),
        )
        source.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 2, 0, 1, 7, 4, 3, 2, 2),
        )
        source.commit()
        source.close()
        sanitize(source_path, output_path, None, False)
        portable = sqlite3.connect(output_path)
        try:
            assert dict(portable.execute("SELECT id, value FROM StringIds")) == {
                1: "cudaLaunchKernel",
                2: "kernel",
            }
            assert portable.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone() == (
                1,
            )
        finally:
            portable.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--verify-file", type=Path, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    for path in args.verify_file:
        verify_no_sensitive_environment(path)
    if args.input is None and args.output is None:
        return
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output must be supplied together")
    sanitize(args.input, args.output, args.source_report, args.force)
    print(f"Portable SQLite: {args.output}")


if __name__ == "__main__":
    main()

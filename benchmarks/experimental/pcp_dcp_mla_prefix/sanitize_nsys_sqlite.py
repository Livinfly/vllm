# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanitize nsys reports and create portable experiment-only SQLite traces."""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import regex as re

MARKER_PREFIX = "VLLM_PCP_DCP_MLA|"
SENSITIVE_NAME_PARTS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)
ENVIRONMENT_NAME = re.compile(rb"^[A-Za-z_][A-Za-z0-9_]*$")
MIN_CREDENTIAL_VALUE_BYTES = 8
NON_CREDENTIAL_ENVIRONMENT_NAMES = {b"PREFIX_TOKENS"}


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
    overlap = (
        max(
            max((len(value) for value in sensitive.values()), default=0),
            1,
        )
        - 1
    )
    tail = b""
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            content = tail + chunk
            leaked.update(name for name, value in sensitive.items() if value in content)
            tail = content[-overlap:] if overlap else b""
    if leaked:
        names = ", ".join(sorted(leaked))
        raise RuntimeError(f"{path} contains sensitive environment values: {names}")


def _is_sensitive_name(name: bytes) -> bool:
    upper_name = name.upper()
    return (
        upper_name not in NON_CREDENTIAL_ENVIRONMENT_NAMES
        and bool(ENVIRONMENT_NAME.fullmatch(name))
        and any(part.encode() in upper_name for part in SENSITIVE_NAME_PARTS)
    )


def _export_report(input_path: Path, output_path: Path, nsys_bin: str) -> None:
    completed = subprocess.run(
        [
            nsys_bin,
            "export",
            "--type=sqlite",
            "--quiet=true",
            "--force-overwrite=true",
            f"--output={output_path}",
            str(input_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nsys export failed ({completed.returncode}): {completed.stderr}"
        )


def _redact_environment_string(value: str) -> tuple[bytes, set[str]]:
    original = value.encode()
    entries = original.split(b"\0")
    redacted_names: set[str] = set()
    for index, entry in enumerate(entries):
        name, separator, environment_value = entry.partition(b"=")
        if (
            not separator
            or not environment_value
            or not _is_sensitive_name(name)
            or set(environment_value) == {ord("*")}
        ):
            continue
        if len(environment_value) < MIN_CREDENTIAL_VALUE_BYTES:
            raise RuntimeError(
                "Refusing to publish a short credential-like environment "
                f"value: {name.decode('ascii', errors='replace')}"
            )
        entries[index] = name + separator + b"*" * len(environment_value)
        redacted_names.add(name.decode("ascii", errors="replace"))
    return b"\0".join(entries), redacted_names


def _report_string_replacements(
    report_path: Path,
    nsys_bin: str,
) -> list[tuple[bytes, bytes, set[str]]]:
    with tempfile.TemporaryDirectory() as directory:
        sqlite_path = Path(directory) / "report.sqlite"
        _export_report(report_path, sqlite_path, nsys_bin)
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            values = connection.execute("SELECT value FROM StringIds").fetchall()
        finally:
            connection.close()

    replacements_by_original: dict[bytes, set[str]] = {}
    for (value,) in values:
        if not isinstance(value, str):
            continue
        for entry in value.encode().split(b"\0"):
            name, separator, environment_value = entry.partition(b"=")
            if (
                not separator
                or not environment_value
                or not _is_sensitive_name(name)
                or set(environment_value) == {ord("*")}
            ):
                continue
            if len(environment_value) < MIN_CREDENTIAL_VALUE_BYTES:
                raise RuntimeError(
                    "Refusing to publish a short credential-like environment "
                    f"value: {name.decode('ascii', errors='replace')}"
                )
            replacements_by_original.setdefault(environment_value, set()).add(
                name.decode("ascii", errors="replace")
            )
    return [
        (original, b"*" * len(original), names)
        for original, names in replacements_by_original.items()
    ]


def sanitize_report(
    input_path: Path,
    output_path: Path,
    redaction_report_path: Path,
    nsys_bin: str,
    force: bool,
) -> None:
    """Redact captured credential-like environment values in an nsys report."""
    if (output_path.exists() or redaction_report_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite report output; pass --force")

    content = input_path.read_bytes()
    replacement_count = 0
    redacted_names: set[str] = set()
    for original, redacted, names in _report_string_replacements(input_path, nsys_bin):
        occurrences = content.count(original)
        if not occurrences:
            raise RuntimeError(
                "Exported environment string was not found in report: "
                f"names={sorted(names)}, bytes={len(original)}"
            )
        content = content.replace(original, redacted)
        replacement_count += occurrences
        redacted_names.update(names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_bytes(content)
        verify_no_sensitive_environment(temporary_path)
        remaining = _report_string_replacements(temporary_path, nsys_bin)
        if remaining:
            names = sorted(
                {name for _, _, row_names in remaining for name in row_names}
            )
            raise RuntimeError(
                "Sanitized report still contains environment values for: "
                + ", ".join(names)
            )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "vllm-nsys-environment-redaction-v2",
        "source_report_sha256": _sha256(input_path),
        "sanitized_report_sha256": _sha256(output_path),
        "report_size_bytes": output_path.stat().st_size,
        "redacted_environment_variables": sorted(redacted_names),
        "redacted_string_occurrences": replacement_count,
        "minimum_credential_value_bytes": MIN_CREDENTIAL_VALUE_BYTES,
        "replacement": "credential-like captured environment values replaced "
        "by same-length ASCII asterisks",
    }
    redaction_report_path.parent.mkdir(parents=True, exist_ok=True)
    redaction_report_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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

        leaked_path = root / "leaked.nsys-rep"
        test_name = "PCP_DCP_TEST_API_KEY"
        test_value = "do-not-publish"
        previous = os.environ.get(test_name)
        os.environ[test_name] = test_value
        leaked_path.write_bytes(test_value.encode())
        try:
            verify_no_sensitive_environment(leaked_path)
        except RuntimeError as error:
            assert test_name in str(error)
        else:
            raise AssertionError("Credential-like value was not detected")
        finally:
            if previous is None:
                del os.environ[test_name]
            else:
                os.environ[test_name] = previous

        redacted, names = _redact_environment_string(
            "HOME=/tmp\0API_KEY=do-not-publish\0EMPTY_SECRET="
        )
        assert redacted == b"HOME=/tmp\0API_KEY=**************\0EMPTY_SECRET="
        assert names == {"API_KEY"}
        assert _is_sensitive_name(b"mixedCaseToken")
        assert not _is_sensitive_name(b"PREFIX_TOKENS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--sanitize-report", type=Path)
    parser.add_argument("--redaction-report", type=Path)
    parser.add_argument("--nsys-bin", default=os.environ.get("NSYS_BIN", "nsys"))
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
    if args.sanitize_report is not None:
        if args.output is None or args.redaction_report is None:
            raise SystemExit(
                "--sanitize-report requires --output and --redaction-report"
            )
        sanitize_report(
            args.sanitize_report,
            args.output,
            args.redaction_report,
            args.nsys_bin,
            args.force,
        )
        print(f"Sanitized nsys report: {args.output}")
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

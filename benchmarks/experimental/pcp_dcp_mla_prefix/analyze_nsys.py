# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze instrumented PCP+DCP MLA ranges in an Nsight Systems report."""

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import regex as re

MARKER_PREFIX = "VLLM_PCP_DCP_MLA"
COMM_CATEGORIES = (
    "context_attention_comm",
    "suffix_cache_comm",
    "hidden_restore_comm",
)
ATTENTION_COMM_CATEGORIES = COMM_CATEGORIES[:2]


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass
class NvtxRange:
    rowid: int
    start: int
    end: int | None
    global_tid: int
    pid: int
    tid: int
    message: str
    fields: dict[str, str]


@dataclass
class RuntimeCall:
    rowid: int
    start: int
    end: int
    global_tid: int
    pid: int
    tid: int
    correlation_id: int
    name: str


@dataclass
class Kernel:
    rowid: int
    start: int
    end: int
    device: int
    context: int
    stream: int
    correlation_id: int
    global_pid: int
    pid: int
    name: str
    short_name: str
    runtime: RuntimeCall | None = None
    marker_ids: set[int] = field(default_factory=set)
    marker_categories: set[str] = field(default_factory=set)

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


def _pid_tid(global_id: int | None) -> tuple[int, int]:
    if global_id is None:
        return 0, 0
    pid = (int(global_id) // 0x1000000) % 0x1000000
    tid = int(global_id) % 0x1000000
    if pid == 0 and int(global_id) < 0x1000000:
        pid = int(global_id)
    return pid, tid


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    ordered = sorted(
        (interval for interval in intervals if interval.end > interval.start),
        key=lambda interval: (interval.start, interval.end),
    )
    merged: list[Interval] = []
    for interval in ordered:
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
            continue
        merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end))
    return merged


def interval_duration(intervals: list[Interval]) -> int:
    return sum(interval.end - interval.start for interval in merge_intervals(intervals))


def intersection_duration(left: list[Interval], right: list[Interval]) -> int:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = 0
    right_index = 0
    duration = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_interval = left_merged[left_index]
        right_interval = right_merged[right_index]
        start = max(left_interval.start, right_interval.start)
        end = min(left_interval.end, right_interval.end)
        duration += max(end - start, 0)
        if left_interval.end <= right_interval.end:
            left_index += 1
        else:
            right_index += 1
    return duration


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(connection):
        return {}
    return dict(connection.execute("SELECT id, value FROM StringIds"))


def _parse_marker(message: str | None) -> dict[str, str] | None:
    if not message or not message.startswith(MARKER_PREFIX + "|"):
        return None
    fields: dict[str, str] = {}
    for component in message.split("|")[1:]:
        if "=" not in component:
            continue
        key, value = component.split("=", 1)
        fields[key] = unquote(value)
    required = {"category", "iteration", "rank"}
    if not required.issubset(fields):
        raise RuntimeError(f"Malformed experiment NVTX marker: {message}")
    return fields


def _load_ranges(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[NvtxRange]:
    tables = _tables(connection)
    if "NVTX_EVENTS" not in tables:
        raise RuntimeError("NVTX_EVENTS is absent; capture with --trace=cuda,nvtx.")
    columns = _columns(connection, "NVTX_EVENTS")
    selected = ["rowid", "start", "end", "globalTid"]
    selected.extend(column for column in ("text", "textId") if column in columns)
    rows = connection.execute(f"SELECT {', '.join(selected)} FROM NVTX_EVENTS")
    ranges = []
    for raw_row in rows:
        row = dict(zip(selected, raw_row, strict=True))
        message = row.get("text")
        if not message and row.get("textId") is not None:
            message = strings.get(row["textId"])
        fields = _parse_marker(message)
        if fields is None:
            continue
        pid, tid = _pid_tid(row["globalTid"])
        end = row["end"]
        ranges.append(
            NvtxRange(
                rowid=row["rowid"],
                start=row["start"],
                end=end if end and end > row["start"] else None,
                global_tid=row["globalTid"],
                pid=pid,
                tid=tid,
                message=message,
                fields=fields,
            )
        )
    if not ranges:
        raise RuntimeError(f"No {MARKER_PREFIX} markers were found.")
    return ranges


def _load_runtime(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[RuntimeCall]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in _tables(connection):
        raise RuntimeError(f"{table} is absent; CUDA tracing was not captured.")
    rows = connection.execute(
        f"SELECT rowid, start, end, globalTid, correlationId, nameId FROM {table}"
    )
    calls = []
    for rowid, start, end, global_tid, correlation_id, name_id in rows:
        pid, tid = _pid_tid(global_tid)
        calls.append(
            RuntimeCall(
                rowid=rowid,
                start=start,
                end=end,
                global_tid=global_tid,
                pid=pid,
                tid=tid,
                correlation_id=correlation_id,
                name=strings.get(name_id, f"string_id:{name_id}"),
            )
        )
    return calls


def _load_kernels(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    runtime_calls: list[RuntimeCall],
) -> list[Kernel]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in _tables(connection):
        raise RuntimeError(f"{table} is absent; no GPU kernels were captured.")
    runtime_by_key: dict[tuple[int, int], list[RuntimeCall]] = defaultdict(list)
    for call in runtime_calls:
        runtime_by_key[(call.pid, call.correlation_id)].append(call)
    rows = connection.execute(
        "SELECT rowid, start, end, deviceId, contextId, streamId, "
        f"correlationId, globalPid, demangledName, shortName FROM {table}"
    )
    kernels = []
    for row in rows:
        (
            rowid,
            start,
            end,
            device,
            context,
            stream,
            correlation_id,
            global_pid,
            demangled_name,
            short_name,
        ) = row
        pid, _ = _pid_tid(global_pid)
        candidates = runtime_by_key.get((pid, correlation_id), [])
        preceding = [call for call in candidates if call.start <= start]
        runtime = (
            max(preceding or candidates, key=lambda call: call.start)
            if candidates
            else None
        )
        kernels.append(
            Kernel(
                rowid=rowid,
                start=start,
                end=end,
                device=device,
                context=context,
                stream=stream,
                correlation_id=correlation_id,
                global_pid=global_pid,
                pid=pid,
                name=strings.get(demangled_name, f"string_id:{demangled_name}"),
                short_name=strings.get(short_name, f"string_id:{short_name}"),
                runtime=runtime,
            )
        )
    return kernels


def _kernels_in_range(marker: NvtxRange, kernels: list[Kernel]) -> list[Kernel]:
    if marker.end is None:
        return []
    return [
        kernel
        for kernel in kernels
        if kernel.runtime is not None
        and kernel.pid == marker.pid
        and kernel.runtime.tid == marker.tid
        and marker.start <= kernel.runtime.start < marker.end
    ]


def _associate_markers(ranges: list[NvtxRange], kernels: list[Kernel]) -> None:
    for marker in ranges:
        for kernel in _kernels_in_range(marker, kernels):
            kernel.marker_ids.add(marker.rowid)
            kernel.marker_categories.add(marker.fields["category"])


def _unique_kernels(kernels: list[Kernel]) -> list[Kernel]:
    return list({kernel.rowid: kernel for kernel in kernels}.values())


def _range_kernels(
    ranges: list[NvtxRange], kernels: list[Kernel], category: str
) -> list[Kernel]:
    matched = []
    for marker in ranges:
        if marker.fields["category"] == category:
            matched.extend(_kernels_in_range(marker, kernels))
    return _unique_kernels(matched)


def _payload(ranges: list[NvtxRange], category: str) -> dict[str, Any]:
    selected = [marker for marker in ranges if marker.fields["category"] == category]
    return {
        "collectives": len(selected),
        "send_bytes": sum(
            int(marker.fields.get("send_bytes", 0)) for marker in selected
        ),
        "recv_bytes": sum(
            int(marker.fields.get("recv_bytes", 0)) for marker in selected
        ),
        "markers": [marker.fields for marker in selected],
    }


def _timing(kernels: list[Kernel]) -> dict[str, Any]:
    intervals = [kernel.interval for kernel in kernels]
    return {
        "kernel_count": len(kernels),
        "union_ns": interval_duration(intervals),
        "raw_sum_ns": sum(interval.end - interval.start for interval in intervals),
    }


def _scope_metrics(
    scope_marker: NvtxRange,
    rank_ranges: list[NvtxRange],
    kernels: list[Kernel],
    nccl_pattern: re.Pattern[str],
) -> tuple[dict[str, Any], list[Kernel]]:
    scope_kernels = _kernels_in_range(scope_marker, kernels)
    if not scope_kernels:
        raise RuntimeError(f"No kernels were associated with {scope_marker.message}")
    compute_kernels = [
        kernel for kernel in scope_kernels if not nccl_pattern.search(kernel.name)
    ]
    scope_kernel_ids = {kernel.rowid for kernel in scope_kernels}
    comm_by_category = {
        category: [
            kernel
            for kernel in _range_kernels(rank_ranges, kernels, category)
            if nccl_pattern.search(kernel.name) and kernel.rowid in scope_kernel_ids
        ]
        for category in ATTENTION_COMM_CATEGORIES
    }
    for category in ATTENTION_COMM_CATEGORIES:
        outside_scope = [
            kernel
            for kernel in _range_kernels(rank_ranges, kernels, category)
            if nccl_pattern.search(kernel.name) and kernel.rowid not in scope_kernel_ids
        ]
        if outside_scope:
            raise RuntimeError(
                f"{category} kernels escaped the {scope_marker.fields['scope']} scope."
            )
    comm_kernels = _unique_kernels(
        [kernel for rows in comm_by_category.values() for kernel in rows]
    )
    scope_nccl = [
        kernel for kernel in scope_kernels if nccl_pattern.search(kernel.name)
    ]
    unclassified = {
        kernel.rowid: kernel
        for kernel in scope_nccl
        if kernel.rowid not in {item.rowid for item in comm_kernels}
    }
    if unclassified:
        names = sorted({kernel.name for kernel in unclassified.values()})
        raise RuntimeError(f"Unclassified NCCL kernels inside attention scope: {names}")

    start = min(kernel.start for kernel in scope_kernels)
    end = max(kernel.end for kernel in scope_kernels)
    total_ns = end - start
    comm_intervals = [kernel.interval for kernel in comm_kernels]
    compute_intervals = [kernel.interval for kernel in compute_kernels]
    comm_ns = interval_duration(comm_intervals)
    compute_ns = interval_duration(compute_intervals)
    overlap_ns = intersection_duration(comm_intervals, compute_intervals)
    exposed_ns = comm_ns - overlap_ns
    if total_ns <= 0 or comm_ns <= 0:
        raise RuntimeError("Scope or communication duration is zero.")
    metrics = {
        "scope": scope_marker.fields["scope"],
        "T_ns": total_ns,
        "C_ns": comm_ns,
        "A_ns": compute_ns,
        "O_ns": overlap_ns,
        "E_ns": exposed_ns,
        "raw_comm_ratio": comm_ns / total_ns,
        "overlap_rate": overlap_ns / comm_ns,
        "exposed_comm_ratio": exposed_ns / total_ns,
        "scope_kernel_count": len(scope_kernels),
        "scope_raw_sum_ns": sum(kernel.end - kernel.start for kernel in scope_kernels),
        "compute_kernel_count": len(compute_kernels),
        "compute_raw_sum_ns": sum(
            kernel.end - kernel.start for kernel in compute_kernels
        ),
        "attention_comm_kernel_count": len(comm_kernels),
        "attention_comm_raw_sum_ns": sum(
            kernel.end - kernel.start for kernel in comm_kernels
        ),
        "categories": {
            category: _timing(category_kernels) | _payload(rank_ranges, category)
            for category, category_kernels in comm_by_category.items()
        },
    }
    return metrics, scope_kernels


def _rank_summary(
    rank: int,
    label: str,
    ranges: list[NvtxRange],
    kernels: list[Kernel],
    nccl_pattern: re.Pattern[str],
) -> dict[str, Any]:
    rank_ranges = [
        marker
        for marker in ranges
        if marker.fields["iteration"] == label and int(marker.fields["rank"]) == rank
    ]
    scope_markers = [
        marker
        for marker in rank_ranges
        if marker.fields["category"] == "scope" and marker.end is not None
    ]
    by_scope: dict[str, list[NvtxRange]] = defaultdict(list)
    for marker in scope_markers:
        by_scope[marker.fields.get("scope", "")].append(marker)
    if set(by_scope) != {"self_attn", "full_layer"} or any(
        len(items) != 1 for items in by_scope.values()
    ):
        raise RuntimeError(f"Expected one self_attn and full_layer range: {by_scope}")

    scope_results = {}
    devices = set()
    for scope, markers in by_scope.items():
        metrics, scope_kernels = _scope_metrics(
            markers[0], rank_ranges, kernels, nccl_pattern
        )
        devices.update(kernel.device for kernel in scope_kernels)
        scope_results[scope] = metrics

    category_results = {}
    for category in COMM_CATEGORIES:
        candidates = _range_kernels(rank_ranges, kernels, category)
        communication = [
            kernel for kernel in candidates if nccl_pattern.search(kernel.name)
        ]
        if not communication:
            candidate_names = sorted({kernel.name for kernel in candidates})
            raise RuntimeError(
                f"No NCCL kernels mapped to rank {rank} {category}; "
                f"candidate kernels: {candidate_names}"
            )
        category_results[category] = _timing(communication) | _payload(
            rank_ranges, category
        )
    context = category_results["context_attention_comm"]
    context_seconds = context["union_ns"] / 1e9
    context["send_only_effective_bandwidth_GBps"] = (
        context["send_bytes"] / context_seconds / 1e9
    )
    attention_path_kernels = _unique_kernels(
        [
            kernel
            for category in ATTENTION_COMM_CATEGORIES
            for kernel in _range_kernels(rank_ranges, kernels, category)
            if nccl_pattern.search(kernel.name)
        ]
    )
    category_results["attention_path_comm"] = _timing(attention_path_kernels) | {
        "collectives": sum(
            category_results[category]["collectives"]
            for category in ATTENTION_COMM_CATEGORIES
        ),
        "send_bytes": sum(
            category_results[category]["send_bytes"]
            for category in ATTENTION_COMM_CATEGORIES
        ),
        "recv_bytes": sum(
            category_results[category]["recv_bytes"]
            for category in ATTENTION_COMM_CATEGORIES
        ),
        "markers": [
            marker
            for category in ATTENTION_COMM_CATEGORIES
            for marker in category_results[category]["markers"]
        ],
    }
    partition = [
        marker.fields
        for marker in rank_ranges
        if marker.fields["category"] == "pcp_partition"
    ]
    return {
        "rank": rank,
        "devices": sorted(devices),
        "scopes": scope_results,
        "communication": category_results,
        "pcp_partition": partition,
    }


def _write_ranges(path: Path, ranges: list[NvtxRange]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["rowid", "pid", "tid", "start_ns", "end_ns", "fields_json", "message"]
        )
        for marker in ranges:
            writer.writerow(
                [
                    marker.rowid,
                    marker.pid,
                    marker.tid,
                    marker.start,
                    marker.end,
                    json.dumps(marker.fields, sort_keys=True),
                    marker.message,
                ]
            )


def _write_kernels(path: Path, kernels: list[Kernel]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rowid",
                "pid",
                "runtime_tid",
                "device",
                "context",
                "stream",
                "start_ns",
                "end_ns",
                "duration_ns",
                "correlation_id",
                "runtime_start_ns",
                "runtime_end_ns",
                "runtime_name",
                "marker_categories",
                "kernel_name",
                "short_name",
            ]
        )
        for kernel in kernels:
            writer.writerow(
                [
                    kernel.rowid,
                    kernel.pid,
                    kernel.runtime.tid if kernel.runtime else None,
                    kernel.device,
                    kernel.context,
                    kernel.stream,
                    kernel.start,
                    kernel.end,
                    kernel.end - kernel.start,
                    kernel.correlation_id,
                    kernel.runtime.start if kernel.runtime else None,
                    kernel.runtime.end if kernel.runtime else None,
                    kernel.runtime.name if kernel.runtime else None,
                    ";".join(sorted(kernel.marker_categories)),
                    kernel.name,
                    kernel.short_name,
                ]
            )


def _write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for rank_result in result["ranks"]:
        for scope, metrics in rank_result["scopes"].items():
            rows.append(
                {
                    "rank": rank_result["rank"],
                    "device": ";".join(map(str, rank_result["devices"])),
                    "scope": scope,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, dict)
                    },
                }
            )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_sqlite(input_path: Path, output_dir: Path) -> Path:
    if input_path.suffix in (".sqlite", ".db"):
        return input_path
    sqlite_path = output_dir / (input_path.name.removesuffix(".nsys-rep") + ".sqlite")
    command = [
        os.environ.get("NSYS_BIN", "nsys"),
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={sqlite_path}",
        str(input_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"nsys export failed ({completed.returncode}): {completed.stderr}"
        )
    return sqlite_path


def analyze(
    input_path: Path,
    output_dir: Path,
    expected_label: str | None,
    expected_ranks: int,
    nccl_regex: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = _export_sqlite(input_path, output_dir)
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        strings = _strings(connection)
        ranges = _load_ranges(connection, strings)
        runtime_calls = _load_runtime(connection, strings)
        kernels = _load_kernels(connection, strings, runtime_calls)
    finally:
        connection.close()
    _associate_markers(ranges, kernels)

    labels = sorted(
        {
            marker.fields["iteration"]
            for marker in ranges
            if marker.fields["category"] == "scope"
        }
    )
    if expected_label is None:
        if len(labels) != 1:
            raise RuntimeError(f"Specify --label; trace contains labels {labels}")
        label = labels[0]
    else:
        label = expected_label
        if label not in labels:
            raise RuntimeError(f"Label {label!r} not found; available labels: {labels}")
    ranks = sorted(
        {
            int(marker.fields["rank"])
            for marker in ranges
            if marker.fields["iteration"] == label
        }
    )
    if ranks != list(range(expected_ranks)):
        raise RuntimeError(f"Expected ranks 0..{expected_ranks - 1}, got {ranks}")
    nccl_pattern = re.compile(nccl_regex)
    rank_results = [
        _rank_summary(rank, label, ranges, kernels, nccl_pattern) for rank in ranks
    ]
    headline = {}
    for scope in ("self_attn", "full_layer"):
        slowest = max(rank_results, key=lambda row: row["scopes"][scope]["T_ns"])
        headline[scope] = {
            "slower_rank": slowest["rank"],
            **slowest["scopes"][scope],
        }
    result = {
        "input": str(input_path),
        "sqlite": str(sqlite_path),
        "label": label,
        "nccl_regex": nccl_regex,
        "ranks": rank_results,
        "headline": headline,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_csv(output_dir / "summary.csv", result)
    _write_ranges(output_dir / "experiment_ranges.csv", ranges)
    _write_kernels(output_dir / "kernels.csv", kernels)
    return result


def _self_test() -> None:
    merged = merge_intervals(
        [Interval(0, 10), Interval(5, 15), Interval(20, 30), Interval(30, 40)]
    )
    assert merged == [Interval(0, 15), Interval(20, 40)]
    assert interval_duration(merged) == 35
    assert (
        intersection_duration(
            [Interval(0, 10), Interval(20, 30)],
            [Interval(5, 25)],
        )
        == 10
    )
    marker = (
        "VLLM_PCP_DCP_MLA|category=scope|iteration=nsys_0|rank=1|"
        "scope=self_attn|activation_dtype=torch.bfloat16"
    )
    parsed = _parse_marker(marker)
    assert parsed is not None
    assert parsed["scope"] == "self_attn"
    assert _pid_tid((123 << 24) | 456) == (123, 456)

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "synthetic.sqlite"
        connection = sqlite3.connect(database)
        connection.executescript(
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
            """
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(1, "cudaLaunchKernel"), (2, "compute_kernel"), (3, "ncclKernel")],
        )
        global_tid = (77 << 24) | 88
        global_pid = 77 << 24
        messages = [
            (0, 100, "scope", "self_attn"),
            (0, 110, "scope", "full_layer"),
            (20, 40, "context_attention_comm", None),
            (50, 60, "suffix_cache_comm", None),
            (120, 130, "hidden_restore_comm", None),
        ]
        for start, end, category, scope in messages:
            fields = f"{MARKER_PREFIX}|category={category}|iteration=nsys_0|rank=0"
            if scope:
                fields += f"|scope={scope}"
            if category != "scope":
                fields += "|send_bytes=16|recv_bytes=16"
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
                (start, end, global_tid, fields),
            )
        runtime_rows = [
            (10, 11, global_tid, 1, 1),
            (25, 26, global_tid, 2, 1),
            (52, 53, global_tid, 3, 1),
            (122, 123, global_tid, 4, 1),
        ]
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            runtime_rows,
        )
        kernel_rows = [
            (1000, 1100, 0, 1, 7, 1, global_pid, 2, 2),
            (1050, 1150, 0, 1, 8, 2, global_pid, 3, 3),
            (1200, 1250, 0, 1, 8, 3, global_pid, 3, 3),
            (1300, 1350, 0, 1, 8, 4, global_pid, 3, 3),
        ]
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            kernel_rows,
        )
        connection.commit()
        connection.close()
        result = analyze(database, Path(directory) / "out", "nsys_0", 1, "nccl")
        assert result["ranks"][0]["scopes"]["self_attn"]["C_ns"] == 150


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("nsys-results"))
    parser.add_argument("--label")
    parser.add_argument("--expected-ranks", type=int, default=2)
    parser.add_argument("--nccl-regex", default=r"(?i)(nccl|msccl)")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if args.input is None:
        raise SystemExit("--input is required unless --self-test is used")
    result = analyze(
        args.input,
        args.output_dir,
        args.label,
        args.expected_ranks,
        args.nccl_regex,
    )
    print(json.dumps(result["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

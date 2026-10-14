from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Configure runtime knobs before importing the native Lance/Vortex engines.
THREADS_ENV = os.getenv("BENCH_THREADS", "")
if THREADS_ENV:
    os.environ["LANCE_CPU_THREADS"] = THREADS_ENV
    os.environ["LANCE_IO_THREADS"] = THREADS_ENV

import lance
import vortex as vx
import vortex.expr as ve

ROOT = Path(os.getenv("DATA_ROOT", "/data"))
PARQUET_DIR = ROOT / "parquet" / "batches" / os.getenv("PARQUET_CODEC_DIR", "zstd")
VORTEX_DIR = ROOT / "vortex" / "batches"
LANCE_DIR = ROOT / "lance" / "batches"
VALIDATION_REPORT = Path(os.getenv("VALIDATION_REPORT", "/data/validation/integrity-report.json"))
OUT_DIR = ROOT / "benchmark"

BATCHES = int(os.getenv("BATCHES", "250"))
ROWS_PER_BATCH = int(os.getenv("ROWS_PER_BATCH", "1000000"))
READ_BATCH_SIZE = int(os.getenv("READ_BATCH_SIZE", "65536"))
WARMUP = int(os.getenv("WARMUP", "1"))
GLOBAL_RUNS = int(os.getenv("RUNS", "0"))
SEED = int(os.getenv("BENCH_SEED", "20261005"))
BENCH_FORMATS = [x.strip().lower() for x in os.getenv("BENCH_FORMATS", "parquet,vortex,lance").split(",") if x.strip()]
CASE_FILTER = os.getenv("CASE", "all").strip()
SINGLE_BATCH = min(max(1, int(os.getenv("SINGLE_BATCH", str((BATCHES + 1) // 2)))), BATCHES)
POINTS = int(os.getenv("POINTS", "128"))

EXPECTED_TOTAL_ROWS = BATCHES * ROWS_PER_BATCH
CATEGORIES = ("analytics", "finance", "iot", "media", "retail", "security", "telemetry", "web")
COUNTRIES = ("DE", "FR", "GB", "US", "CA", "JP", "CN", "IN", "BR", "AU", "ES", "IT", "NL", "SE", "PL", "CH")
STATUSES = ("ok", "created", "updated", "retry", "failed")

@dataclass
class Metrics:
    wall_s: float
    cpu_user_s: float
    cpu_system_s: float
    peak_rss_mib: float
    rchar_bytes: int | None
    read_bytes: int | None

@dataclass
class Result:
    format: str
    case: str
    iteration: int
    warmup: bool
    batch_scope: str
    files: int
    input_footprint_bytes: int
    input_rows: int
    output_rows: int
    expected_output_rows: int
    rows_per_s: float
    logical_footprint_mib_per_s: float
    wall_s: float
    cpu_user_s: float
    cpu_system_s: float
    cpu_total_s: float
    cpu_parallelism: float
    peak_rss_mib: float
    kernel_rchar_bytes: int | None
    kernel_read_bytes: int | None
    thread_setting: str
    status: str
    error: str | None = None


class ValidationError(RuntimeError):
    pass


class PeakRSS:
    def __init__(self, period: float = 0.05) -> None:
        self.period = period
        self._stop = threading.Event()
        self.peak = 0
        self.thread: threading.Thread | None = None

    @staticmethod
    def rss_bytes() -> int:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except (FileNotFoundError, OSError, ValueError):
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.rss_bytes())
            self._stop.wait(self.period)

    def __enter__(self) -> "PeakRSS":
        self.peak = self.rss_bytes()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.peak = max(self.peak, self.rss_bytes())


def proc_io() -> tuple[int | None, int | None]:
    try:
        values: dict[str, int] = {}
        with open("/proc/self/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key.strip()] = int(value.strip())
        return values.get("rchar"), values.get("read_bytes")
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def now_cpu() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def time_call(fn: Callable[[], int]) -> tuple[int, Metrics]:
    gc.collect()
    user0, sys0 = now_cpu()
    rchar0, read0 = proc_io()
    started = time.perf_counter()
    with PeakRSS() as peak:
        output = fn()
    wall = time.perf_counter() - started
    user1, sys1 = now_cpu()
    rchar1, read1 = proc_io()
    return output, Metrics(
        wall_s=wall,
        cpu_user_s=user1 - user0,
        cpu_system_s=sys1 - sys0,
        peak_rss_mib=peak.peak / (1024 * 1024),
        rchar_bytes=(rchar1 - rchar0) if rchar0 is not None and rchar1 is not None else None,
        read_bytes=(read1 - read0) if read0 is not None and read1 is not None else None,
    )


def log(tag: str, message: str) -> None:
    print(f"[{tag:<9}] {message}", flush=True)


def expected_name(batch: int) -> str:
    return f"batch_{batch:06d}"


def parquet_files() -> list[Path]:
    return [PARQUET_DIR / f"{expected_name(i)}.parquet" for i in range(1, BATCHES + 1)]


def vortex_files() -> list[Path]:
    return [VORTEX_DIR / f"{expected_name(i)}.vortex" for i in range(1, BATCHES + 1)]


def lance_dirs() -> list[Path]:
    return [LANCE_DIR / f"{expected_name(i)}.lance" for i in range(1, BATCHES + 1)]


def footprint(paths: Sequence[Path]) -> tuple[int, int]:
    total = 0
    files = 0
    for root in paths:
        if root.is_file():
            total += root.stat().st_size
            files += 1
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
                    files += 1
    return total, files


def check_integrity_gate() -> dict[str, object]:
    if not VALIDATION_REPORT.is_file():
        raise ValidationError(f"missing integrity report: {VALIDATION_REPORT}")
    report = json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))
    if report.get("result") != "passed":
        raise ValidationError("integrity report is not passed")
    if int(report.get("batches_verified", -1)) != BATCHES:
        raise ValidationError("integrity report batch count mismatch")
    if int(report.get("total_rows_verified", -1)) != EXPECTED_TOTAL_ROWS:
        raise ValidationError("integrity report row count mismatch")
    return report


def check_inventories() -> None:
    for name, paths, suffix in (
        ("Parquet", parquet_files(), ".parquet"),
        ("Vortex", vortex_files(), ".vortex"),
        ("Lance", lance_dirs(), ".lance"),
    ):
        if len(paths) != BATCHES:
            raise ValidationError(f"{name}: expected {BATCHES} paths")
        for i, path in enumerate(paths, 1):
            if suffix == ".lance":
                if not path.is_dir():
                    raise ValidationError(f"missing Lance dataset: {path}")
            elif not path.is_file():
                raise ValidationError(f"missing {name} file: {path}")


def setup_threads() -> None:
    if THREADS_ENV:
        threads = int(THREADS_ENV)
        if threads <= 0:
            raise ValidationError("BENCH_THREADS must be positive")
        pa.set_cpu_count(threads)
        try:
            vx.set_worker_threads(threads)
        except Exception:
            pass


def consume(batches: Iterable[pa.RecordBatch]) -> int:
    rows = 0
    for batch in batches:
        rows += batch.num_rows
        _ = batch.nbytes
    return rows


def user_id_range(pct: float) -> tuple[int, int, int]:
    domain = 10_000_000
    width = int(domain * pct)
    return domain - width, domain, width


def expected_filter_rows(pct: float) -> int:
    return BATCHES * int(ROWS_PER_BATCH * pct)


def expected_category_rows() -> int:
    return BATCHES * ROWS_PER_BATCH // len(CATEGORIES)

def expected_country_rows() -> int:
    return BATCHES * ROWS_PER_BATCH // len(COUNTRIES)

def expected_status_rows() -> int:
    return BATCHES * ROWS_PER_BATCH // len(STATUSES)


def representative_random_ids() -> list[int]:
    rng = random.Random(SEED)
    start = (SINGLE_BATCH - 1) * ROWS_PER_BATCH
    return sorted(rng.sample(range(start, start + ROWS_PER_BATCH), min(POINTS, ROWS_PER_BATCH)))


def parquet_scan(paths: Sequence[Path], columns: list[str] | None, filter_expr: ds.Expression | None) -> int:
    total = 0
    for path in paths:
        dataset = ds.dataset(str(path), format="parquet")
        batches = dataset.to_batches(
            columns=columns,
            filter=filter_expr,
            batch_size=READ_BATCH_SIZE,
            use_threads=True,
        )
        for batch in batches:
            total += batch.num_rows
            _ = batch.nbytes
    return total


def vortex_scan(paths: Sequence[Path], columns: list[str] | None, expr: object | None) -> int:
    total = 0
    for path in paths:
        vf = vx.open(str(path))
        reader = vf.to_arrow(
            projection=columns,
            expr=expr,
            batch_size=READ_BATCH_SIZE,
        )
        for batch in reader:
            total += batch.num_rows
            _ = batch.nbytes
    return total


def lance_scan(paths: Sequence[Path], columns: list[str] | None, filter_sql: str | None) -> int:
    total = 0
    for path in paths:
        dataset = lance.dataset(str(path))
        batches = dataset.to_batches(
            columns=columns,
            filter=filter_sql,
            batch_size=READ_BATCH_SIZE,
        )
        for batch in batches:
            total += batch.num_rows
            _ = batch.nbytes
    return total


def metadata_open(fmt: str, paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        if fmt == "parquet":
            pf = pq.ParquetFile(path)
            _ = pf.schema_arrow
        elif fmt == "vortex":
            vf = vx.open(str(path))
            _ = vf.dtype
        else:
            ds_ = lance.dataset(str(path))
            _ = ds_.schema
        count += 1
    return count


def metadata_count(fmt: str, paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        if fmt == "parquet":
            meta = pq.ParquetFile(path).metadata
            if meta is None:
                raise ValidationError(f"missing Parquet metadata: {path}")
            total += meta.num_rows
        elif fmt == "vortex":
            total += len(vx.open(str(path)))
        else:
            total += lance.dataset(str(path)).count_rows()
    return total


def native_random(fmt: str, path: Path, global_ids: list[int]) -> int:
    columns = ["id", "user_id", "value", "category"]

    if fmt == "lance":
        # Lance `take()` uses positional row indices within the dataset, not the
        # logical value of our `id` column. The selected benchmark batch contains
        # contiguous global ids, so translate global ids -> local row positions.
        batch_start = (SINGLE_BATCH - 1) * ROWS_PER_BATCH
        batch_end = batch_start + ROWS_PER_BATCH
        local_ids = [row_id - batch_start for row_id in global_ids]
        if any(index < 0 or index >= ROWS_PER_BATCH for index in local_ids):
            raise ValidationError(
                f"invalid Lance random-access indices: batch range="
                f"[{batch_start}, {batch_end}), global_ids={global_ids[:5]}"
            )
        table = lance.dataset(str(path)).take(local_ids, columns=columns)
    else:
        if fmt == "parquet":
            arrow_ids = pa.array(global_ids, type=pa.int64())
            dataset = ds.dataset(str(path), format="parquet")
            expr = ds.field("id").isin(arrow_ids)
            table = dataset.to_table(columns=columns, filter=expr, use_threads=True)
        else:
            # Vortex provides a native positional take() API. Avoid translating
            # random access into an Arrow `isin(...)` predicate: some Vortex
            # versions cannot lower that Substrait list expression.
            batch_start = (SINGLE_BATCH - 1) * ROWS_PER_BATCH
            batch_end = batch_start + ROWS_PER_BATCH
            local_ids = [row_id - batch_start for row_id in global_ids]
            if any(index < 0 or index >= ROWS_PER_BATCH for index in local_ids):
                raise ValidationError(
                    f"invalid Vortex random-access indices: batch range="
                    f"[{batch_start}, {batch_end}), global_ids={global_ids[:5]}"
                )
            dataset = vx.open(str(path)).to_dataset()
            table = dataset.take(
                pa.array(local_ids, type=pa.int64()),
                columns=columns,
                use_threads=True,
            )

    actual_ids = table.column("id").to_pylist()
    expected_ids = sorted(global_ids)
    if sorted(actual_ids) != expected_ids:
        raise ValidationError(
            f"random-access mismatch for {fmt}: expected ids {expected_ids[:8]}... "
            f"got {sorted(actual_ids)[:8]}..."
        )
    if table.num_rows != len(global_ids):
        raise ValidationError(
            f"random-access row count mismatch for {fmt}: "
            f"expected={len(global_ids)} actual={table.num_rows}"
        )
    return table.num_rows


def make_case(name: str, description: str, files_mode: str, default_runs: int, expected: int | Callable[[], int]) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "files_mode": files_mode,
        "default_runs": default_runs,
        "expected": expected,
    }


def cases() -> list[dict[str, object]]:
    return [
        make_case("metadata_open_250", "open every dataset/file and materialize schema metadata only", "all", 3, BATCHES),
        make_case("metadata_count_250", "open every dataset/file and obtain row counts", "all", 3, EXPECTED_TOTAL_ROWS),
        make_case("full_scan_1_batch", "full scan of one representative batch", "single", 2, ROWS_PER_BATCH),
        make_case("full_scan_250", "full scan of all 250 batches", "all", 1, EXPECTED_TOTAL_ROWS),
        make_case("projection_1col_250", "scan only high-cardinality user_id column", "all", 2, EXPECTED_TOTAL_ROWS),
        make_case("projection_3col_250", "scan user_id/value/category", "all", 2, EXPECTED_TOTAL_ROWS),
        make_case("filter_user_id_1pct_250", "high-cardinality user_id range, 1% selectivity", "all", 2, expected_filter_rows(0.01)),
        make_case("filter_user_id_10pct_250", "high-cardinality user_id range, 10% selectivity", "all", 2, expected_filter_rows(0.10)),
        make_case("filter_user_id_50pct_250", "high-cardinality user_id range, 50% selectivity", "all", 2, expected_filter_rows(0.50)),
        make_case("filter_value_1pct_250", "value >= 199, approximately 1% selectivity", "all", 2, expected_filter_rows(0.01)),
        make_case("filter_category_12_5pct_250", "category = analytics, exactly 1/8 of rows", "all", 2, expected_category_rows()),
        make_case("filter_country_6_25pct_250", "country = DE, exactly 1/16 of rows", "all", 2, expected_country_rows()),
        make_case("filter_status_20pct_250", "status = ok, exactly 1/5 of rows", "all", 2, expected_status_rows()),
        make_case("random_access_128_one_batch", "128 random rows: native take where available, predicate fallback for Parquet/Vortex", "single", 5, POINTS),
    ]


def case_paths(fmt: str, files_mode: str) -> list[Path]:
    paths = {"parquet": parquet_files(), "vortex": vortex_files(), "lance": lance_dirs()}[fmt]
    return [paths[SINGLE_BATCH - 1]] if files_mode == "single" else paths


def run_case(fmt: str, case_name: str) -> tuple[int, int, str]:
    paths = case_paths(fmt, next(c["files_mode"] for c in cases() if c["name"] == case_name))
    if case_name == "metadata_open_250":
        return metadata_open(fmt, paths), BATCHES, "all"
    if case_name == "metadata_count_250":
        return metadata_count(fmt, paths), EXPECTED_TOTAL_ROWS, "all"
    if case_name == "full_scan_1_batch":
        if fmt == "parquet":
            rows = parquet_scan(paths, None, None)
        elif fmt == "vortex":
            rows = vortex_scan(paths, None, None)
        else:
            rows = lance_scan(paths, None, None)
        return rows, ROWS_PER_BATCH, "single"
    if case_name == "full_scan_250":
        if fmt == "parquet":
            rows = parquet_scan(paths, None, None)
        elif fmt == "vortex":
            rows = vortex_scan(paths, None, None)
        else:
            rows = lance_scan(paths, None, None)
        return rows, EXPECTED_TOTAL_ROWS, "all"
    if case_name == "projection_1col_250":
        if fmt == "parquet":
            rows = parquet_scan(paths, ["user_id"], None)
        elif fmt == "vortex":
            rows = vortex_scan(paths, ["user_id"], None)
        else:
            rows = lance_scan(paths, ["user_id"], None)
        return rows, EXPECTED_TOTAL_ROWS, "all"
    if case_name == "projection_3col_250":
        cols = ["user_id", "value", "category"]
        if fmt == "parquet":
            rows = parquet_scan(paths, cols, None)
        elif fmt == "vortex":
            rows = vortex_scan(paths, cols, None)
        else:
            rows = lance_scan(paths, cols, None)
        return rows, EXPECTED_TOTAL_ROWS, "all"
    if case_name.startswith("filter_user_id_"):
        pct = {"filter_user_id_1pct_250": 0.01, "filter_user_id_10pct_250": 0.10, "filter_user_id_50pct_250": 0.50}[case_name]
        total = 0
        for batch_num, path in enumerate(paths, 1):
            low, high, _ = user_id_range(pct)
            # case_paths(all) always returns batch 1..BATCHES; single isn't used for these cases.
            if fmt == "parquet":
                expr = (ds.field("user_id") >= low) & (ds.field("user_id") < high)
                total += parquet_scan([path], ["user_id", "value"], expr)
            elif fmt == "vortex":
                expr = (ve.column("user_id") >= low) & (ve.column("user_id") < high)
                total += vortex_scan([path], ["user_id", "value"], expr)
            else:
                filt = f"user_id >= {low} AND user_id < {high}"
                total += lance_scan([path], ["user_id", "value"], filt)
        return total, expected_filter_rows(pct), "all"
    if case_name == "filter_value_1pct_250":
        total = 0
        for path in paths:
            if fmt == "parquet":
                expr = ds.field("value") >= 199.0
                total += parquet_scan([path], ["user_id", "value"], expr)
            elif fmt == "vortex":
                expr = ve.column("value") >= 199.0
                total += vortex_scan([path], ["user_id", "value"], expr)
            else:
                total += lance_scan([path], ["id", "value"], "value >= 199.0")
        return total, expected_filter_rows(0.01), "all"
    if case_name == "filter_category_12_5pct_250":
        total = 0
        for path in paths:
            if fmt == "parquet":
                expr = ds.field("category") == "analytics"
                total += parquet_scan([path], ["id", "category"], expr)
            elif fmt == "vortex":
                expr = ve.column("category") == "analytics"
                total += vortex_scan([path], ["id", "category"], expr)
            else:
                total += lance_scan([path], ["id", "category"], "category = 'analytics'")
        return total, expected_category_rows(), "all"
    if case_name == "filter_country_6_25pct_250":
        total = 0
        for path in paths:
            if fmt == "parquet":
                total += parquet_scan([path], ["id", "country"], ds.field("country") == "DE")
            elif fmt == "vortex":
                total += vortex_scan([path], ["id", "country"], ve.column("country") == "DE")
            else:
                total += lance_scan([path], ["id", "country"], "country = 'DE'")
        return total, expected_country_rows(), "all"
    if case_name == "filter_status_20pct_250":
        total = 0
        for path in paths:
            if fmt == "parquet":
                total += parquet_scan([path], ["id", "status"], ds.field("status") == "ok")
            elif fmt == "vortex":
                total += vortex_scan([path], ["id", "status"], ve.column("status") == "ok")
            else:
                total += lance_scan([path], ["id", "status"], "status = 'ok'")
        return total, expected_status_rows(), "all"
    if case_name == "random_access_128_one_batch":
        ids = representative_random_ids()
        rows = native_random(fmt, paths[0], ids)
        return rows, len(ids), "single"
    raise KeyError(case_name)


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in raw:
        if row["warmup"] or row["status"] != "ok":
            continue
        grouped.setdefault((str(row["format"]), str(row["case"])), []).append(row)
    summary: list[dict[str, object]] = []
    for (fmt, case), rows in sorted(grouped.items()):
        walls = [float(r["wall_s"]) for r in rows]
        rps = [float(r["rows_per_s"]) for r in rows]
        rss = [float(r["peak_rss_mib"]) for r in rows]
        summary.append({
            "format": fmt,
            "case": case,
            "runs": len(rows),
            "median_wall_s": statistics.median(walls),
            "min_wall_s": min(walls),
            "max_wall_s": max(walls),
            "median_rows_per_s": statistics.median(rps),
            "median_peak_rss_mib": statistics.median(rss),
            "expected_rows": rows[0]["expected_output_rows"],
            "status": "ok" if all(r["output_rows"] == r["expected_output_rows"] for r in rows) else "failed",
        })
    return summary


def write_outputs(raw: list[dict[str, object]], summary: list[dict[str, object]], meta: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(
        json.dumps({"metadata": meta, "runs": raw, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if raw:
        with (OUT_DIR / "benchmark-runs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
            writer.writeheader()
            writer.writerows(raw)
    if summary:
        with (OUT_DIR / "benchmark-summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    lines = [
        "# Benchmark summary",
        "",
        f"Formats: {', '.join(BENCH_FORMATS)}",
        f"Rows: {EXPECTED_TOTAL_ROWS:,}",
        f"Read batch size: {READ_BATCH_SIZE}",
        f"Threads: {THREADS_ENV or 'library default'}",
        "",
        "| Format | Case | Median s | Rows/s | Peak RSS MiB | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['format']} | {row['case']} | {row['median_wall_s']:.4f} | "
            f"{row['median_rows_per_s']:.0f} | {row['median_peak_rss_mib']:.1f} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Methodology",
        "",
        "This benchmark starts only after a passed Parquet/Vortex/Lance integrity report. "
        "Each case consumes Arrow record batches without collecting the full 250M-row dataset in memory.",
        "",
        "The benchmark uses format-native projection/filter/random-access APIs. Vortex and Lance can "
        "push down projection and predicates; Lance also exposes native `take()` for random rows. "
        "Those capabilities are intentionally tested as part of the format comparison.",
        "",
        "The benchmark does not flush the host OS page cache. Results should therefore be described as "
        "local filesystem warm/cached measurements unless the experiment is repeated in a separately "
        "rebooted or cache-controlled environment.",
    ])
    (OUT_DIR / "benchmark-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_threads()
    log("start", "Parquet / Vortex / Lance comprehensive benchmark")
    log("config", f"batches={BATCHES} rows_per_batch={ROWS_PER_BATCH} total_rows={EXPECTED_TOTAL_ROWS:,}")
    log("config", f"read_batch_size={READ_BATCH_SIZE} warmup={WARMUP} global_runs={GLOBAL_RUNS or 'case defaults'} threads={THREADS_ENV or 'library default'}")
    log("source", f"parquet={PARQUET_DIR}")
    log("source", f"vortex ={VORTEX_DIR}")
    log("source", f"lance  ={LANCE_DIR}")

    integrity = check_integrity_gate()
    check_inventories()
    log("gate", f"integrity PASSED: {integrity.get('total_rows_verified'):,} rows across {integrity.get('batches_verified')} batches")

    footprints: dict[str, dict[str, int]] = {}
    for fmt, paths in (("parquet", parquet_files()), ("vortex", vortex_files()), ("lance", lance_dirs())):
        bytes_total, physical_files = footprint(paths)
        footprints[fmt] = {"bytes": bytes_total, "physical_files": physical_files}
        log("size", f"{fmt}: {bytes_total / (1024**3):.3f} GiB, physical_files={physical_files}")

    selected_cases = [c for c in cases() if CASE_FILTER == "all" or c["name"] == CASE_FILTER]
    if not selected_cases:
        raise SystemExit(f"unknown CASE={CASE_FILTER!r}")

    raw: list[dict[str, object]] = []
    rng = random.Random(SEED)
    for case in selected_cases:
        case_name = str(case["name"])
        runs = GLOBAL_RUNS if GLOBAL_RUNS > 0 else int(case["default_runs"])
        order = BENCH_FORMATS.copy()
        rng.shuffle(order)
        for fmt in order:
            paths = case_paths(fmt, str(case["files_mode"]))
            input_bytes = sum(footprint([p])[0] for p in paths)
            log("case", f"{case_name} / {fmt} / runs={runs} input={input_bytes/(1024**3):.3f} GiB")
            for warm in range(WARMUP):
                try:
                    output_rows, expected_rows, scope = run_case(fmt, case_name)
                    if output_rows != expected_rows:
                        raise ValidationError(f"warmup output rows mismatch: expected={expected_rows} got={output_rows}")
                    log("warmup", f"{case_name} / {fmt}: rows={output_rows:,} OK")
                except Exception as exc:
                    log("FAIL", f"warmup {case_name}/{fmt}: {type(exc).__name__}: {exc}")
                    raise

            for iteration in range(1, runs + 1):
                started = time.perf_counter()
                try:
                    output_rows, expected_rows, scope = None, int(case["expected"]) if isinstance(case["expected"], int) else 0, str(case["files_mode"])
                    def _run() -> int:
                        nonlocal output_rows, expected_rows, scope
                        output_rows, expected_rows, scope = run_case(fmt, case_name)
                        return int(output_rows)
                    output_rows, metrics = time_call(_run)
                    expected_rows = int(expected_rows)
                    status = "ok" if output_rows == expected_rows else "failed"
                    error = None if status == "ok" else f"expected {expected_rows}, got {output_rows}"
                except Exception as exc:
                    metrics = Metrics(0,0,0,0,None,None)
                    output_rows = -1
                    expected_rows = int(case["expected"]) if isinstance(case["expected"], int) else -1
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                wall = metrics.wall_s
                cpu_total = metrics.cpu_user_s + metrics.cpu_system_s
                row = Result(
                    format=fmt,
                    case=case_name,
                    iteration=iteration,
                    warmup=False,
                    batch_scope=str(case["files_mode"]),
                    files=len(paths),
                    input_footprint_bytes=input_bytes,
                    input_rows=ROWS_PER_BATCH if case["files_mode"] == "single" else EXPECTED_TOTAL_ROWS,
                    output_rows=int(output_rows),
                    expected_output_rows=expected_rows,
                    rows_per_s=(float(output_rows) / wall) if wall > 0 and output_rows >= 0 else 0.0,
                    logical_footprint_mib_per_s=(input_bytes / (1024*1024) / wall) if wall > 0 else 0.0,
                    wall_s=wall,
                    cpu_user_s=metrics.cpu_user_s,
                    cpu_system_s=metrics.cpu_system_s,
                    cpu_total_s=cpu_total,
                    cpu_parallelism=(cpu_total / wall) if wall > 0 else 0.0,
                    peak_rss_mib=metrics.peak_rss_mib,
                    kernel_rchar_bytes=metrics.rchar_bytes,
                    kernel_read_bytes=metrics.read_bytes,
                    thread_setting=THREADS_ENV or "library_default",
                    status=status,
                    error=error,
                )
                raw.append(asdict(row))
                log("run", f"{case_name} / {fmt} #{iteration}: {status} {wall:.4f}s rows={output_rows:,} rows/s={row.rows_per_s:,.0f}")
                if status != "ok":
                    raise SystemExit(error or "benchmark failed")

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "pyarrow": pa.__version__,
        "vortex": getattr(vx, "__version__", "unknown"),
        "lance": getattr(lance, "__version__", "unknown"),
        "threads": THREADS_ENV or "library_default",
        "read_batch_size": READ_BATCH_SIZE,
        "warmup": WARMUP,
        "seed": SEED,
        "batches": BATCHES,
        "rows_per_batch": ROWS_PER_BATCH,
        "total_rows": EXPECTED_TOTAL_ROWS,
        "single_batch": SINGLE_BATCH,
        "points": POINTS,
        "integrity_report": str(VALIDATION_REPORT),
        "integrity_report_elapsed_s": integrity.get("elapsed_s"),
        "footprints": footprints,
        "cache_control": "none; host OS page cache not flushed",
        "vortex_workers": vx.worker_threads() if hasattr(vx, "worker_threads") else None,
    }
    summary = summarize(raw)
    write_outputs(raw, summary, meta)
    log("report", str(OUT_DIR / "benchmark.json"))
    log("report", str(OUT_DIR / "benchmark-summary.csv"))
    log("report", str(OUT_DIR / "benchmark-summary.md"))
    log("result", f"PASSED cases={len(selected_cases)} formats={','.join(BENCH_FORMATS)} runs={len(raw)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import vortex as vx

PARQUET_DIR = Path(os.getenv("PARQUET_INPUT_DIR", "/data/parquet/batches/zstd"))
VORTEX_DIR = Path(os.getenv("VORTEX_INPUT_DIR", "/data/vortex/batches"))
LANCE_DIR = Path(os.getenv("LANCE_INPUT_DIR", "/data/lance/batches"))
REPORT_DIR = Path(os.getenv("VALIDATION_OUTPUT_DIR", "/data/validation"))
BATCHES = int(os.getenv("BATCHES", "250"))
ROWS_PER_BATCH = int(os.getenv("ROWS_PER_BATCH", "1000000"))
READ_BATCH_SIZE = int(os.getenv("READ_BATCH_SIZE", "65536"))
FAIL_FAST = os.getenv("FAIL_FAST", "false").lower() in {"1", "true", "yes"}
VORTEX_MAX_THREADS = os.getenv("VORTEX_MAX_THREADS")


class ValidationError(RuntimeError):
    pass


def log(prefix: str, message: str) -> None:
    print(f"[{prefix:<8}] {message}", flush=True)


def die(message: str) -> None:
    log("error", message)
    raise SystemExit(1)


def expected_name(batch: int) -> str:
    return f"batch_{batch:06d}"


def check_exact_inventory(directory: Path, suffix: str, expected_count: int) -> None:
    if not directory.is_dir():
        raise ValidationError(f"missing directory: {directory}")

    if suffix == ".lance":
        actual = sorted(
            p.name for p in directory.iterdir()
            if p.is_dir() and p.name.endswith(suffix)
        )
    else:
        actual = sorted(
            p.name for p in directory.iterdir()
            if p.is_file() and p.name.endswith(suffix)
        )

    expected = [f"{expected_name(i)}{suffix}" for i in range(1, expected_count + 1)]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValidationError(
            f"inventory mismatch in {directory}: missing={missing[:10]} extra={extra[:10]}"
        )


def parquet_stream(path: Path) -> tuple[pa.Schema, int, Iterator[pa.RecordBatch]]:
    pf = pq.ParquetFile(path)
    if pf.metadata is None:
        raise ValidationError(f"Parquet metadata is missing: {path}")
    schema = pf.schema_arrow
    rows = pf.metadata.num_rows
    batches = pf.iter_batches(batch_size=READ_BATCH_SIZE, use_threads=True)
    return schema, rows, batches


def vortex_stream(path: Path, schema: pa.Schema) -> tuple[pa.Schema, int, Iterator[pa.RecordBatch]]:
    vf = vx.open(str(path))
    rows = len(vf)
    reader = vf.to_arrow(batch_size=READ_BATCH_SIZE, schema=schema)
    return reader.schema, rows, iter(reader)


def lance_stream(path: Path) -> tuple[pa.Schema, int, Iterator[pa.RecordBatch]]:
    ds = lance.dataset(str(path))
    rows = ds.count_rows()
    schema = ds.schema
    # Current Lance Python API does not expose use_threads on LanceDataset.scanner().
    # to_batches() is the streaming API and accepts batch_size directly.
    batches = ds.to_batches(batch_size=READ_BATCH_SIZE)
    return schema, rows, iter(batches)


class AlignedBatchStream:
    def __init__(self, batches: Iterator[pa.RecordBatch], label: str) -> None:
        self.batches = iter(batches)
        self.label = label
        self.current: pa.RecordBatch | None = None
        self.offset = 0
        self.exhausted = False

    def _ensure_current(self) -> None:
        if self.current is not None and self.offset < self.current.num_rows:
            return
        try:
            self.current = next(self.batches)
            self.offset = 0
        except StopIteration:
            self.current = None
            self.offset = 0
            self.exhausted = True

    def remaining(self) -> int:
        self._ensure_current()
        if self.current is None:
            return 0
        return self.current.num_rows - self.offset

    def take(self, count: int) -> pa.RecordBatch:
        self._ensure_current()
        if self.current is None:
            raise ValidationError(f"unexpected end of {self.label} stream")
        if count <= 0 or count > self.current.num_rows - self.offset:
            raise ValueError("invalid aligned batch size")
        result = self.current.slice(self.offset, count)
        self.offset += count
        return result

    def is_finished(self) -> bool:
        self._ensure_current()
        return self.exhausted and self.current is None


def validate_schema(reference: pa.Schema, candidate: pa.Schema, label: str) -> None:
    if not reference.equals(candidate, check_metadata=True):
        raise ValidationError(
            f"schema mismatch: parquet vs {label}\n"
            f"parquet={reference}\n"
            f"{label}={candidate}"
        )


def compare_record_batches(
    reference: pa.RecordBatch,
    candidate: pa.RecordBatch,
    label: str,
    global_row: int,
) -> None:
    if reference.num_columns != candidate.num_columns:
        raise ValidationError(
            f"column count mismatch at row {global_row}: parquet={reference.num_columns} {label}={candidate.num_columns}"
        )
    for column_index, (ref_array, cand_array) in enumerate(
        zip(reference.columns, candidate.columns, strict=True)
    ):
        if not ref_array.equals(cand_array):
            name = reference.schema.field(column_index).name
            raise ValidationError(
                f"data mismatch in {label}: column={name!r} column_index={column_index} "
                f"starting_row={global_row} rows={reference.num_rows}"
            )


def compare_streams(
    parquet_batches: Iterator[pa.RecordBatch],
    vortex_batches: Iterator[pa.RecordBatch],
    lance_batches: Iterator[pa.RecordBatch],
    total_rows: int,
) -> int:
    streams = [
        AlignedBatchStream(parquet_batches, "Parquet"),
        AlignedBatchStream(vortex_batches, "Vortex"),
        AlignedBatchStream(lance_batches, "Lance"),
    ]
    verified = 0

    while verified < total_rows:
        remaining = [stream.remaining() for stream in streams]
        if any(value == 0 for value in remaining):
            labels = ", ".join(
                f"{label}={value}"
                for label, value in zip(("Parquet", "Vortex", "Lance"), remaining, strict=True)
            )
            raise ValidationError(
                f"unexpected end-of-stream at row {verified}: {labels}"
            )

        count = min(remaining)
        reference = streams[0].take(count)
        vortex_batch = streams[1].take(count)
        lance_batch = streams[2].take(count)

        if not reference.schema.equals(vortex_batch.schema, check_metadata=True):
            raise ValidationError(f"Arrow schema changed during Vortex scan at row {verified}")
        if not reference.schema.equals(lance_batch.schema, check_metadata=True):
            raise ValidationError(f"Arrow schema changed during Lance scan at row {verified}")

        compare_record_batches(reference, vortex_batch, "Vortex", verified)
        compare_record_batches(reference, lance_batch, "Lance", verified)
        verified += count

    for stream in streams:
        if not stream.is_finished():
            # There are extra rows beyond the declared total.
            raise ValidationError(f"extra rows detected after row {total_rows} in {stream.label}")

    return verified


def validate_batch(batch: int) -> dict[str, object]:
    stem = expected_name(batch)
    parquet_path = PARQUET_DIR / f"{stem}.parquet"
    vortex_path = VORTEX_DIR / f"{stem}.vortex"
    lance_path = LANCE_DIR / f"{stem}.lance"

    if not parquet_path.is_file():
        raise ValidationError(f"missing Parquet file: {parquet_path}")
    if not vortex_path.is_file():
        raise ValidationError(f"missing Vortex file: {vortex_path}")
    if not lance_path.is_dir():
        raise ValidationError(f"missing Lance dataset: {lance_path}")

    parquet_schema, parquet_rows, parquet_batches = parquet_stream(parquet_path)
    if parquet_rows != ROWS_PER_BATCH:
        raise ValidationError(
            f"Parquet row count mismatch: expected={ROWS_PER_BATCH} got={parquet_rows}"
        )

    vortex_schema, vortex_rows, vortex_batches = vortex_stream(vortex_path, parquet_schema)
    lance_schema, lance_rows, lance_batches = lance_stream(lance_path)

    if vortex_rows != ROWS_PER_BATCH:
        raise ValidationError(
            f"Vortex row count mismatch: expected={ROWS_PER_BATCH} got={vortex_rows}"
        )
    if lance_rows != ROWS_PER_BATCH:
        raise ValidationError(
            f"Lance row count mismatch: expected={ROWS_PER_BATCH} got={lance_rows}"
        )

    validate_schema(parquet_schema, vortex_schema, "Vortex")
    validate_schema(parquet_schema, lance_schema, "Lance")

    verified_rows = compare_streams(
        parquet_batches,
        vortex_batches,
        lance_batches,
        ROWS_PER_BATCH,
    )

    schema_description = {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in parquet_schema
        ],
        "metadata": {
            (key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)): (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else str(value)
            )
            for key, value in (parquet_schema.metadata or {}).items()
        },
    }

    return {
        "batch": batch,
        "rows": verified_rows,
        "schema": schema_description,
        "parquet_bytes": parquet_path.stat().st_size,
        "vortex_bytes": vortex_path.stat().st_size,
        "lance_bytes": sum(
            p.stat().st_size for p in lance_path.rglob("*") if p.is_file()
        ),
        "status": "ok",
    }


def main() -> None:
    if VORTEX_MAX_THREADS:
        vx.set_worker_threads(int(VORTEX_MAX_THREADS))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    log("start", "Parquet / Vortex / Lance integrity & consistency check")
    log("config", f"batches={BATCHES} rows_per_batch={ROWS_PER_BATCH} read_batch_size={READ_BATCH_SIZE}")
    log("source", f"parquet={PARQUET_DIR}")
    log("source", f"vortex ={VORTEX_DIR}")
    log("source", f"lance  ={LANCE_DIR}")

    try:
        check_exact_inventory(PARQUET_DIR, ".parquet", BATCHES)
        check_exact_inventory(VORTEX_DIR, ".vortex", BATCHES)
        check_exact_inventory(LANCE_DIR, ".lance", BATCHES)
        log("files", "exact batch inventories: OK")
    except ValidationError as exc:
        die(str(exc))

    for batch in range(1, BATCHES + 1):
        started = time.perf_counter()
        try:
            result = validate_batch(batch)
            results.append(result)
            elapsed = time.perf_counter() - started
            log(
                "batch",
                f"{batch:3d}/{BATCHES} rows={ROWS_PER_BATCH} "
                f"verified=OK elapsed={elapsed:.3f}s",
            )
        except Exception as exc:  # deliberate: collect all failures for an audit report
            elapsed = time.perf_counter() - started
            failure = {
                "batch": batch,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": elapsed,
            }
            failures.append(failure)
            log("FAIL", f"{batch:3d}/{BATCHES} {failure['error']}")
            if FAIL_FAST:
                raise SystemExit(2) from exc

    elapsed_all = time.perf_counter() - started_all
    report = {
        "check": "parquet-vortex-lance-integrity",
        "result": "passed" if not failures else "failed",
        "batches_expected": BATCHES,
        "batches_verified": len(results),
        "rows_per_batch": ROWS_PER_BATCH,
        "total_rows_verified": sum(int(x["rows"]) for x in results),
        "read_batch_size": READ_BATCH_SIZE,
        "vortex_version": getattr(vx, "__version__", None),
        "lance_version": getattr(lance, "__version__", None),
        "elapsed_s": elapsed_all,
        "files": results,
        "failures": failures,
    }
    report_path = REPORT_DIR / "integrity-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("report", str(report_path))
    if failures:
        log("result", f"FAILED batches={len(failures)}; see integrity-report.json")
        raise SystemExit(2)

    log("result", f"PASSED files={BATCHES} rows={BATCHES * ROWS_PER_BATCH:,} "
                   f"elapsed={elapsed_all:.3f}s")
    log("ok", "all data values, schemas and files are consistent across Parquet/Vortex/Lance")


if __name__ == "__main__":
    main()

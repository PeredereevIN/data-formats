from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import vortex as vx


INPUT_DIR = Path(os.getenv("PARQUET_INPUT_DIR", "/data/parquet/batches/zstd"))
OUTPUT_DIR = Path(os.getenv("VORTEX_OUTPUT_DIR", "/data/vortex/batches"))
BATCHES = int(os.getenv("BATCHES", "250"))
ROWS_PER_BATCH = int(os.getenv("ROWS_PER_BATCH", "1000000"))
VORTEX_STRATEGY = os.getenv("VORTEX_STRATEGY", "btrblocks").lower()
VORTEX_MAX_THREADS = os.getenv("VORTEX_MAX_THREADS")
OVERWRITE = os.getenv("VORTEX_OVERWRITE", "true").lower() in {"1", "true", "yes"}
VALIDATE = os.getenv("VORTEX_VALIDATE", "true").lower() in {"1", "true", "yes"}
BATCH_SIZE = int(os.getenv("READ_BATCH_SIZE", "65536"))
CONVERTER_WORKERS = max(1, int(os.getenv("CONVERTER_WORKERS", "2")))


def die(message: str) -> None:
    print(f"[error ] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def writer_options():
    if VORTEX_STRATEGY in {"btrblocks", "default"}:
        return vx.io.VortexWriteOptions.default()
    if VORTEX_STRATEGY == "compact":
        return vx.io.VortexWriteOptions.compact()
    die(f"Unsupported VORTEX_STRATEGY={VORTEX_STRATEGY}; use btrblocks or compact")


def expected_name(batch: int) -> str:
    return f"batch_{batch:06d}"


def convert_batch(batch: int, options) -> dict[str, object]:
    stem = expected_name(batch)
    source = INPUT_DIR / f"{stem}.parquet"
    target = OUTPUT_DIR / f"{stem}.vortex"
    if not source.is_file():
        raise RuntimeError(f"missing Parquet source: {source}")
    if target.exists():
        if not OVERWRITE:
            raise RuntimeError(f"Vortex output already exists: {target}")
        target.unlink()

    started = time.perf_counter()
    parquet_file = pq.ParquetFile(source)
    if parquet_file.metadata is None:
        raise RuntimeError(f"missing Parquet metadata: {source}")
    source_rows = parquet_file.metadata.num_rows
    if source_rows != ROWS_PER_BATCH:
        raise RuntimeError(f"unexpected row count in {source.name}: expected={ROWS_PER_BATCH}, got={source_rows}")
    reader = pa.RecordBatchReader.from_batches(
        parquet_file.schema_arrow,
        parquet_file.iter_batches(batch_size=BATCH_SIZE, use_threads=True),
    )
    options.write(reader, str(target))
    elapsed = time.perf_counter() - started
    target_bytes = target.stat().st_size
    if VALIDATE:
        actual_rows = len(vx.open(str(target)))
        if actual_rows != ROWS_PER_BATCH:
            raise RuntimeError(f"Vortex validation failed for {target.name}: expected={ROWS_PER_BATCH}, got={actual_rows}")
    return {
        "batch": batch, "rows": source_rows, "source": str(source), "target": str(target),
        "source_bytes": source.stat().st_size, "target_bytes": target_bytes,
        "elapsed_s": elapsed, "throughput_mib_s": target_bytes / (1024 * 1024) / elapsed if elapsed else 0.0,
        "strategy": VORTEX_STRATEGY, "validated": VALIDATE,
    }


def main() -> None:
    if VORTEX_MAX_THREADS:
        vx.set_worker_threads(int(VORTEX_MAX_THREADS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    options = writer_options()
    print("vortex-format-study converter", flush=True)
    print(f"vortex={getattr(vx, '__version__', 'unknown')}", flush=True)
    print(f"source={INPUT_DIR}", flush=True)
    print(f"target={OUTPUT_DIR}", flush=True)
    print(f"batches={BATCHES} rows_per_batch={ROWS_PER_BATCH} strategy={VORTEX_STRATEGY} read_batch_size={BATCH_SIZE} workers={CONVERTER_WORKERS}", flush=True)

    manifest: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=CONVERTER_WORKERS) as pool:
        futures = {pool.submit(convert_batch, batch, options): batch for batch in range(1, BATCHES + 1)}
        for completed, future in enumerate(as_completed(futures), 1):
            batch = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                for f in futures:
                    f.cancel()
                die(f"batch {batch}: {type(exc).__name__}: {exc}")
            manifest.append(result)
            print(f"[batch ] {batch:3d}/{BATCHES} rows={result['rows']} size={int(result['target_bytes'])/(1024*1024):.2f} MiB convert={float(result['elapsed_s']):.3f} s", flush=True)
            if completed == BATCHES or completed % max(1, BATCHES // 20) == 0:
                done_bytes = sum(int(x["target_bytes"]) for x in manifest)
                print(f"          progress={completed}/{BATCHES} cumulative_size={done_bytes/(1024**3):.3f} GiB", flush=True)

    manifest.sort(key=lambda x: int(x["batch"]))
    total_bytes = sum(int(x["target_bytes"]) for x in manifest)
    total_rows = sum(int(x["rows"]) for x in manifest)
    elapsed_all = time.perf_counter() - started_all
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps({
        "format": "vortex", "version": getattr(vx, "__version__", None), "strategy": VORTEX_STRATEGY,
        "batches": BATCHES, "rows_per_batch": ROWS_PER_BATCH, "workers": CONVERTER_WORKERS,
        "total_rows": total_rows, "total_bytes": total_bytes, "elapsed_s": elapsed_all,
        "throughput_gib_s": total_bytes/(1024**3)/elapsed_all if elapsed_all else 0.0,
        "validated": VALIDATE, "files": manifest,
    }, indent=2), encoding="utf-8")
    print(f"[total ] files={BATCHES} rows={total_rows} size={total_bytes/(1024**3):.3f} GiB elapsed={elapsed_all:.3f} s throughput={total_bytes/(1024**3)/elapsed_all:.3f} GiB/s", flush=True)
    print(f"[manifest] {OUTPUT_DIR / 'manifest.json'}", flush=True)
    print("[ok    ] Parquet -> Vortex conversion completed", flush=True)


if __name__ == "__main__":
    main()

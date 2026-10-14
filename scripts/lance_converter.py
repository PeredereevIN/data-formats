from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lance
import pyarrow.parquet as pq

INPUT_DIR = Path(os.getenv("PARQUET_INPUT_DIR", "/data/parquet/batches/zstd"))
OUTPUT_DIR = Path(os.getenv("LANCE_OUTPUT_DIR", "/data/lance/batches"))
BATCHES = int(os.getenv("BATCHES", "250"))
ROWS_PER_BATCH = int(os.getenv("ROWS_PER_BATCH", "1000000"))
READ_BATCH_SIZE = int(os.getenv("READ_BATCH_SIZE", "65536"))
MAX_ROWS_PER_GROUP = int(os.getenv("LANCE_MAX_ROWS_PER_GROUP", "128000"))
MAX_ROWS_PER_FILE = int(os.getenv("LANCE_MAX_ROWS_PER_FILE", str(ROWS_PER_BATCH)))
STORAGE_VERSION = os.getenv("LANCE_STORAGE_VERSION", "2.2")
OVERWRITE = os.getenv("LANCE_OVERWRITE", "true").lower() in {"1", "true", "yes"}
VALIDATE = os.getenv("LANCE_VALIDATE", "true").lower() in {"1", "true", "yes"}
CONVERTER_WORKERS = max(1, int(os.getenv("CONVERTER_WORKERS", "2")))


def die(message: str) -> None:
    print(f"[error ] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def expected_name(batch: int) -> str:
    return f"batch_{batch:06d}"


def convert_batch(batch: int) -> dict[str, object]:
    stem = expected_name(batch)
    source = INPUT_DIR / f"{stem}.parquet"
    target = OUTPUT_DIR / f"{stem}.lance"
    if not source.is_file():
        raise RuntimeError(f"missing Parquet source: {source}")
    if target.exists():
        if not OVERWRITE:
            raise RuntimeError(f"Lance output already exists: {target}")
        shutil.rmtree(target)

    started = time.perf_counter()
    parquet_file = pq.ParquetFile(source)
    if parquet_file.metadata is None:
        raise RuntimeError(f"missing Parquet metadata: {source}")
    source_rows = parquet_file.metadata.num_rows
    if source_rows != ROWS_PER_BATCH:
        raise RuntimeError(f"unexpected row count in {source.name}: expected={ROWS_PER_BATCH}, got={source_rows}")
    schema = parquet_file.schema_arrow
    batches = parquet_file.iter_batches(batch_size=READ_BATCH_SIZE, use_threads=True)
    lance.write_dataset(
        batches, str(target), schema=schema, mode="overwrite",
        data_storage_version=STORAGE_VERSION,
        max_rows_per_group=MAX_ROWS_PER_GROUP,
        max_rows_per_file=MAX_ROWS_PER_FILE,
    )
    elapsed = time.perf_counter() - started
    target_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    actual_rows = None
    if VALIDATE:
        reopened = lance.dataset(str(target))
        actual_rows = reopened.count_rows()
        if actual_rows != ROWS_PER_BATCH:
            raise RuntimeError(f"Lance validation failed for {target.name}: expected={ROWS_PER_BATCH}, got={actual_rows}")
        if reopened.schema != schema:
            raise RuntimeError(f"Lance schema mismatch for {target.name}")
    return {
        "batch": batch, "rows": source_rows, "source": str(source), "target": str(target),
        "source_bytes": source.stat().st_size, "target_bytes": target_bytes,
        "elapsed_s": elapsed, "throughput_mib_s": target_bytes/(1024*1024)/elapsed if elapsed else 0.0,
        "data_storage_version": STORAGE_VERSION, "max_rows_per_group": MAX_ROWS_PER_GROUP,
        "max_rows_per_file": MAX_ROWS_PER_FILE, "validated": VALIDATE, "validated_rows": actual_rows,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    print("lance-format-study converter", flush=True)
    print(f"lance={getattr(lance, '__version__', 'unknown')}", flush=True)
    print(f"source={INPUT_DIR}", flush=True)
    print(f"target={OUTPUT_DIR}", flush=True)
    print(f"batches={BATCHES} rows_per_batch={ROWS_PER_BATCH} storage_version={STORAGE_VERSION} read_batch_size={READ_BATCH_SIZE} max_rows_per_group={MAX_ROWS_PER_GROUP} max_rows_per_file={MAX_ROWS_PER_FILE} workers={CONVERTER_WORKERS}", flush=True)
    if STORAGE_VERSION not in {"2.0", "2.1", "2.2"}:
        die("LANCE_STORAGE_VERSION must be 2.0, 2.1 or 2.2")

    manifest: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=CONVERTER_WORKERS) as pool:
        futures = {pool.submit(convert_batch, batch): batch for batch in range(1, BATCHES + 1)}
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
        "format": "lance", "version": getattr(lance, "__version__", None),
        "data_storage_version": STORAGE_VERSION, "batches": BATCHES, "rows_per_batch": ROWS_PER_BATCH,
        "workers": CONVERTER_WORKERS, "max_rows_per_group": MAX_ROWS_PER_GROUP,
        "max_rows_per_file": MAX_ROWS_PER_FILE, "total_rows": total_rows, "total_bytes": total_bytes,
        "elapsed_s": elapsed_all, "throughput_gib_s": total_bytes/(1024**3)/elapsed_all if elapsed_all else 0.0,
        "validated": VALIDATE, "files": manifest,
    }, indent=2), encoding="utf-8")
    print(f"[total ] files={BATCHES} rows={total_rows} size={total_bytes/(1024**3):.3f} GiB elapsed={elapsed_all:.3f} s throughput={total_bytes/(1024**3)/elapsed_all:.3f} GiB/s", flush=True)
    print(f"[manifest] {OUTPUT_DIR / 'manifest.json'}", flush=True)
    print("[ok    ] Parquet -> Lance conversion completed", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze benchmark results and generate tables/charts for the report.

Input:
  data/benchmark/benchmark.json   (preferred)
  or data/benchmark/benchmark-summary.csv

Output:
  CSV tables, Markdown report and matplotlib PNG charts.

No benchmark execution happens here: this module only analyzes existing results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FORMAT_ORDER = ["parquet", "vortex", "lance"]
CASE_ORDER = [
    "metadata_open_250",
    "metadata_count_250",
    "full_scan_1_batch",
    "full_scan_250",
    "projection_1col_250",
    "projection_3col_250",
    "filter_user_id_1pct_250",
    "filter_user_id_10pct_250",
    "filter_user_id_50pct_250",
    "filter_value_1pct_250",
    "filter_category_12_5pct_250",
    "filter_country_6_25pct_250",
    "filter_status_20pct_250",
    "random_access_128_one_batch",
]
CASE_LABELS = {
    "metadata_open_250": "Metadata open (250)",
    "metadata_count_250": "Metadata count (250)",
    "full_scan_1_batch": "Full scan (1 batch)",
    "full_scan_250": "Full scan (250 batches)",
    "projection_1col_250": "Projection (1 column)",
    "projection_3col_250": "Projection (3 columns)",
    "filter_user_id_1pct_250": "Filter user_id (1%)",
    "filter_user_id_10pct_250": "Filter user_id (10%)",
    "filter_user_id_50pct_250": "Filter user_id (50%)",
    "filter_value_1pct_250": "Filter value (1%)",
    "filter_category_12_5pct_250": "Filter category (12.5%)",
    "filter_country_6_25pct_250": "Filter country (6.25%)",
    "filter_status_20pct_250": "Filter status (20%)",
    "random_access_128_one_batch": "Random access (128)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_INPUT", "/data/benchmark/benchmark.json")),
        help="benchmark.json or benchmark-summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("ANALYSIS_OUTPUT", "/data/analysis")),
        help="directory for generated analysis artifacts",
    )
    return parser.parse_args()


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload, payload.get("summary", [])

    if path.suffix.lower() == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for key in (
                    "runs",
                    "median_wall_s",
                    "min_wall_s",
                    "max_wall_s",
                    "median_rows_per_s",
                    "median_peak_rss_mib",
                    "expected_rows",
                ):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                rows.append(row)
        return {"summary": rows}, rows

    raise ValueError("Input must be .json or .csv")


def ordered_formats(values: list[str]) -> list[str]:
    return [fmt for fmt in FORMAT_ORDER if fmt in values] + [
        fmt for fmt in values if fmt not in FORMAT_ORDER
    ]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def load_raw_runs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("runs", []) or []


def derive_raw_metric(raw_runs: list[dict[str, Any]], case: str, fmt: str, field: str) -> float:
    values = [
        float(r[field])
        for r in raw_runs
        if r.get("case") == case and r.get("format") == fmt and r.get("status") == "ok" and r.get(field) is not None
    ]
    return median(values)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.{digits}f}"


def fmt_gib(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def main() -> int:
    args = parse_args()
    payload, summary = load_results(args.input)
    out = args.output_dir
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    summary = [row for row in summary if row.get("status", "ok") == "ok"]
    if not summary:
        raise RuntimeError("No successful summary rows found")

    raw_runs = load_raw_runs(payload)
    formats = ordered_formats(sorted({str(row["format"]) for row in summary}))
    raw_cases = list(dict.fromkeys(str(row["case"]) for row in summary))
    cases = [case for case in CASE_ORDER if case in raw_cases] + [case for case in raw_cases if case not in CASE_ORDER]
    by_key = {(str(r["format"]), str(r["case"])): r for r in summary}

    footprints: dict[str, dict[str, float]] = {}
    for fmt in formats:
        fp = payload.get("metadata", {}).get("footprints", {}).get(fmt, {})
        if fp:
            footprints[fmt] = {
                "bytes": float(fp.get("bytes", 0)),
                "gib": fmt_gib(float(fp.get("bytes", 0))),
                "physical_files": float(fp.get("physical_files", 0)),
            }

    storage_rows = []
    baseline_bytes = footprints.get("parquet", {}).get("bytes")
    for fmt in formats:
        info = footprints.get(fmt, {})
        storage_rows.append(
            {
                "format": fmt,
                "bytes": info.get("bytes", math.nan),
                "gib": info.get("gib", math.nan),
                "physical_files": info.get("physical_files", math.nan),
                "vs_parquet_ratio": info.get("bytes", math.nan) / baseline_bytes if baseline_bytes else math.nan,
                "space_change_vs_parquet_pct": (info.get("bytes", math.nan) / baseline_bytes - 1) * 100 if baseline_bytes else math.nan,
            }
        )
    write_csv(
        out / "storage_analysis.csv",
        ["format", "bytes", "gib", "physical_files", "vs_parquet_ratio", "space_change_vs_parquet_pct"],
        storage_rows,
    )

    # Core per-case comparison.
    speedup_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    for case in cases:
        baseline = by_key.get(("parquet", case))
        if not baseline:
            continue
        parquet_time = float(baseline["median_wall_s"])
        timings = {
            fmt: float(by_key[(fmt, case)]["median_wall_s"])
            for fmt in formats
            if (fmt, case) in by_key
        }
        winner = min(timings, key=timings.get)
        winner_rows.append({"case": case, "winner": winner, "parquet_s": parquet_time})
        for fmt in formats:
            row = by_key.get((fmt, case))
            if not row:
                continue
            wall = float(row["median_wall_s"])
            speedup_vs_parquet = parquet_time / wall if wall else math.nan
            speedup_rows.append(
                {
                    "case": case,
                    "format": fmt,
                    "median_wall_s": wall,
                    "rows_per_s": float(row["median_rows_per_s"]),
                    "peak_rss_mib": float(row["median_peak_rss_mib"]),
                    "speedup_vs_parquet": speedup_vs_parquet,
                    "time_change_vs_parquet_pct": (wall / parquet_time - 1) * 100 if parquet_time else math.nan,
                    "winner": fmt == winner,
                }
            )

    write_csv(
        out / "speedup_vs_parquet.csv",
        [
            "case",
            "format",
            "median_wall_s",
            "rows_per_s",
            "peak_rss_mib",
            "speedup_vs_parquet",
            "time_change_vs_parquet_pct",
            "winner",
        ],
        speedup_rows,
    )
    write_csv(out / "workload_winners.csv", ["case", "winner", "parquet_s"], winner_rows)

    # Winner counts.
    counts = defaultdict(int)
    for row in winner_rows:
        counts[row["winner"]] += 1
    winner_count_rows = [{"format": fmt, "wins": counts[fmt]} for fmt in formats]
    write_csv(out / "winner_counts.csv", ["format", "wins"], winner_count_rows)

    # Resource comparison.
    resource_rows = []
    for fmt in formats:
        fmt_rows = [r for r in summary if r["format"] == fmt]
        resource_rows.append(
            {
                "format": fmt,
                "mean_peak_rss_mib": statistics.mean(float(r["median_peak_rss_mib"]) for r in fmt_rows),
                "max_peak_rss_mib": max(float(r["median_peak_rss_mib"]) for r in fmt_rows),
            }
        )
    write_csv(out / "resource_summary.csv", ["format", "mean_peak_rss_mib", "max_peak_rss_mib"], resource_rows)

    # ---------- Charts ----------
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180})

    # 1. Storage footprint.
    if storage_rows:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        vals = [r["gib"] for r in storage_rows]
        labels = [r["format"].capitalize() for r in storage_rows]
        bars = ax.bar(labels, vals)
        ax.set_title("Physical storage footprint")
        ax.set_ylabel("GiB")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(plots / "01_storage_footprint.png", bbox_inches="tight")
        plt.close(fig)

    # 2. Physical file count.
    if storage_rows:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        vals = [r["physical_files"] for r in storage_rows]
        labels = [r["format"].capitalize() for r in storage_rows]
        bars = ax.bar(labels, vals)
        ax.set_title("Physical file count")
        ax.set_ylabel("Files")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{int(value):,}".replace(",", " "), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(plots / "02_physical_files.png", bbox_inches="tight")
        plt.close(fig)

    # 3. Wall time across workloads, logarithmic scale because the ranges differ strongly.
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cases))
    width = 0.8 / max(1, len(formats))
    for i, fmt in enumerate(formats):
        vals = [float(by_key[(fmt, case)]["median_wall_s"]) if (fmt, case) in by_key else math.nan for case in cases]
        ax.bar(x + (i - (len(formats)-1)/2)*width, vals, width=width, label=fmt.capitalize())
    ax.set_title("Median wall time by workload")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(case, case) for case in cases], rotation=40, ha="right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "03_wall_time_by_workload.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Rows/s throughput.
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, fmt in enumerate(formats):
        vals = [float(by_key[(fmt, case)]["median_rows_per_s"]) if (fmt, case) in by_key else math.nan for case in cases]
        ax.bar(x + (i - (len(formats)-1)/2)*width, vals, width=width, label=fmt.capitalize())
    ax.set_title("Median throughput by workload")
    ax.set_ylabel("Rows / second (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(case, case) for case in cases], rotation=40, ha="right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "04_throughput_by_workload.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Speedup of every format vs Parquet.
    non_baseline = [fmt for fmt in formats if fmt != "parquet"]
    if non_baseline:
        fig, ax = plt.subplots(figsize=(14, 7))
        offset_width = 0.8 / max(1, len(non_baseline))
        for i, fmt in enumerate(non_baseline):
            vals = []
            for case in cases:
                row = by_key.get((fmt, case))
                base = by_key.get(("parquet", case))
                vals.append(float(base["median_wall_s"]) / float(row["median_wall_s"]) if row and base else math.nan)
            ax.bar(x + (i - (len(non_baseline)-1)/2)*offset_width, vals, width=offset_width, label=f"{fmt.capitalize()} / Parquet")
        ax.axhline(1.0, linewidth=1)
        ax.set_title("Speedup relative to Parquet")
        ax.set_ylabel("Speedup (×)")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LABELS.get(case, case) for case in cases], rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "05_speedup_vs_parquet.png", bbox_inches="tight")
        plt.close(fig)

    # 6. Peak RSS.
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, fmt in enumerate(formats):
        vals = [float(by_key[(fmt, case)]["median_peak_rss_mib"]) if (fmt, case) in by_key else math.nan for case in cases]
        ax.bar(x + (i - (len(formats)-1)/2)*width, vals, width=width, label=fmt.capitalize())
    ax.set_title("Peak RSS by workload")
    ax.set_ylabel("MiB")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(case, case) for case in cases], rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "06_peak_rss_by_workload.png", bbox_inches="tight")
    plt.close(fig)

    # 7. Winner count.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["format"].capitalize() for r in winner_count_rows]
    vals = [r["wins"] for r in winner_count_rows]
    bars = ax.bar(labels, vals)
    ax.set_title("Workload wins")
    ax.set_ylabel("Number of winning workloads")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(plots / "07_workload_wins.png", bbox_inches="tight")
    plt.close(fig)

    # 8. Category-level overview: normalized time vs best format.
    overview_rows = []
    for case in cases:
        entries = [
            (fmt, float(by_key[(fmt, case)]["median_wall_s"]))
            for fmt in formats if (fmt, case) in by_key
        ]
        best = min(value for _, value in entries)
        for fmt, value in entries:
            overview_rows.append({"case": case, "format": fmt, "relative_to_best": value / best})
    write_csv(out / "relative_to_best.csv", ["case", "format", "relative_to_best"], overview_rows)

    # Markdown report.
    report_lines = [
        "# Benchmark analytics report",
        "",
        f"Input: `{args.input}`",
        "",
        "## Storage",
        "",
        "| Format | GiB | Physical files | vs Parquet | Size change vs Parquet |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in storage_rows:
        report_lines.append(
            f"| {row['format']} | {row['gib']:.3f} | {int(row['physical_files'])} | "
            f"{row['vs_parquet_ratio']:.3f}× | {row['space_change_vs_parquet_pct']:+.1f}% |"
        )

    report_lines += [
        "",
        "## Workload winners",
        "",
        "| Workload | Winner | Parquet time, s |",
        "|---|---|---:|",
    ]
    for row in winner_rows:
        report_lines.append(f"| {CASE_LABELS.get(row['case'], row['case'])} | {row['winner']} | {row['parquet_s']:.4f} |")

    report_lines += ["", "## Winner count", ""]
    for row in winner_count_rows:
        report_lines.append(f"- **{row['format']}**: {row['wins']} workloads")

    report_lines += ["", "## Strongest speedups vs Parquet", ""]
    for fmt in non_baseline:
        candidates = [r for r in speedup_rows if r["format"] == fmt]
        candidates.sort(key=lambda r: r["speedup_vs_parquet"], reverse=True)
        report_lines.append(f"### {fmt.capitalize()}")
        for row in candidates[:3]:
            report_lines.append(
                f"- `{CASE_LABELS.get(row['case'], row['case'])}`: **{row['speedup_vs_parquet']:.2f}×** "
                f"({row['median_wall_s']:.4f} s vs {by_key[("parquet", row['case'])]['median_wall_s']:.4f} s Parquet)"
            )

    report_lines += [
        "",
        "## Generated charts",
        "",
        "- `plots/01_storage_footprint.png`",
        "- `plots/02_physical_files.png`",
        "- `plots/03_wall_time_by_workload.png`",
        "- `plots/04_throughput_by_workload.png`",
        "- `plots/05_speedup_vs_parquet.png`",
        "- `plots/06_peak_rss_by_workload.png`",
        "- `plots/07_workload_wins.png`",
        "",
        "## Interpretation notes",
        "",
        "- Speedup is calculated as `Parquet median wall time / format median wall time`.",
        "- Values above 1× are faster than Parquet; values below 1× are slower.",
        "- The benchmark's original cache-control and random-access methodological caveats remain applicable.",
    ]
    (out / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    with (out / "analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input": str(args.input),
                "formats": formats,
                "cases": cases,
                "storage": storage_rows,
                "winner_counts": winner_count_rows,
                "generated_plots": sorted(p.name for p in plots.glob("*.png")),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ok] analytics written to {out}")
    print(f"[ok] plots: {len(list(plots.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize a saved env trace file: mean metrics, mean reward, and the effect of ICD-10 normalization.

    python scripts/display_report.py results/traces/env_qwen3-4b_patient_diagnosis_heldout.renormalized.jsonl
    python scripts/display_report.py results/traces/*.jsonl --top 20

Needs no database. Run scripts/renormalize_traces.py first to get the "before normalization"
side of the comparison: it keeps the pre-normalization numbers in meta.reward_original /
meta.metrics_original. Traces without them are reported as a plain summary. Only
patient_diagnosis episodes are renormalized, so the comparison covers just those; the mean
metrics and reward cover every episode in the file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

EPS = 1e-9


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def mean_metrics(metric_dicts: list[dict | None]) -> dict[str, float]:
    """Mean of every numeric metric over the episodes that report it."""
    vals: dict[str, list[float]] = defaultdict(list)
    for m in metric_dicts:
        for k, v in (m or {}).items():
            if (v := _num(v)) is not None:
                vals[k].append(v)
    return {k: mean(v) for k, v in sorted(vals.items())}


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs and len(xs) else None


def build_report(records: list[dict]) -> dict:
    rewards = [r["reward"] for r in records if _num(r.get("reward")) is not None]
    report = {
        "episodes": len(records),
        "tasks": sorted({(r.get("meta") or {}).get("task") or "?" for r in records}),
        "mean_reward": mean(rewards),
        "mean_metrics": mean_metrics([r.get("metrics") for r in records]),
        "normalization": None,
    }

    # Episodes that went through renormalize_traces.py.
    pairs = [r for r in records if "reward_original" in (r.get("meta") or {})]
    if not pairs:
        return report

    rows = []
    for r in pairs:
        meta = r["meta"]
        old, new = _num(meta["reward_original"]) or 0.0, _num(r.get("reward")) or 0.0
        rows.append({"gt_id": meta.get("gt_id"), "old": old, "new": new, "delta": new - old, "record": r})

    old_m = mean_metrics([r["meta"].get("metrics_original") for r in pairs])
    new_m = mean_metrics([r.get("metrics") for r in pairs])
    report["normalization"] = {
        "episodes": len(rows),
        "mean_reward_before": mean([x["old"] for x in rows]),
        "mean_reward_after": mean([x["new"] for x in rows]),
        "improved": sum(x["delta"] > EPS for x in rows),
        "regressed": sum(x["delta"] < -EPS for x in rows),
        "unchanged": sum(abs(x["delta"]) <= EPS for x in rows),
        "metric_deltas": {k: (old_m.get(k), new_m[k]) for k in new_m},
        "codes_corrected": sum(r["meta"].get("icd10_corrected") or 0 for r in pairs),
        "codes_resolved": sum(r["meta"].get("icd10_resolved") or 0 for r in pairs),
        "codes_resolved_snomed": sum(r["meta"].get("icd10_resolved_snomed") or 0 for r in pairs),
        "regressions": sorted((x for x in rows if x["delta"] < -EPS), key=lambda x: x["delta"]),
    }
    return report


def _fmt(x: float | None) -> str:
    return "   n/a " if x is None else f"{x:7.4f}"


def print_report(path: Path, rep: dict, top: int) -> None:
    print(f"\n{'=' * 78}\n{path}\n{'=' * 78}")
    print(f"episodes: {rep['episodes']}   tasks: {', '.join(rep['tasks'])}")
    print(f"mean reward: {_fmt(rep['mean_reward']).strip()}")

    print("\nMean metrics")
    for k, v in rep["mean_metrics"].items():
        print(f"  {k:<36}{v:8.4f}")
    if not rep["mean_metrics"]:
        print("  (none recorded)")

    norm = rep["normalization"]
    print("\nICD-10 normalization")
    if norm is None:
        print("  no meta.reward_original in this trace; run scripts/renormalize_traces.py to compare")
        return
    before, after = norm["mean_reward_before"], norm["mean_reward_after"]
    print(f"  patient_diagnosis episodes compared: {norm['episodes']}")
    print(f"  mean reward: {before:.4f} -> {after:.4f}  ({after - before:+.4f})")
    print(f"  improved {norm['improved']}   regressed {norm['regressed']}   unchanged {norm['unchanged']}")
    print(f"  codes corrected {norm['codes_corrected']}, resolved from name {norm['codes_resolved']} "
          f"({norm['codes_resolved_snomed']} via SNOMED)")

    print(f"\n  {'metric':<36}{'before':>8}{'after':>9}{'change':>9}")
    for k, (b, a) in norm["metric_deltas"].items():
        change = "" if b is None else f"{a - b:+9.4f}"
        print(f"  {k:<36}{_fmt(b)} {_fmt(a)}{change}")

    regs = norm["regressions"]
    if not regs:
        print("\n  No regressions: no episode's reward dropped.")
        return
    print(f"\n  Regressions ({len(regs)}), worst first{f', showing {top}' if len(regs) > top else ''}")
    print(f"  {'gt_id':>8}{'before':>9}{'after':>9}{'change':>9}  corrected/resolved")
    for x in regs[:top]:
        meta = x["record"]["meta"]
        fixes = f"{meta.get('icd10_corrected', 0)}/{meta.get('icd10_resolved', 0)}"
        print(f"  {x['gt_id']!s:>8}{x['old']:9.4f}{x['new']:9.4f}{x['delta']:+9.4f}  {fixes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+", type=Path, help="trace .jsonl files")
    ap.add_argument("--top", type=int, default=10, help="max regressions to list per file (default 10)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON instead")
    args = ap.parse_args()

    for path in args.traces:
        rep = build_report(load_records(path))
        if args.json:
            if rep["normalization"]:
                for x in rep["normalization"]["regressions"]:
                    x.pop("record")
            print(json.dumps({"file": str(path), **rep}, indent=2))
        else:
            print_report(path, rep, args.top)


if __name__ == "__main__":
    main()

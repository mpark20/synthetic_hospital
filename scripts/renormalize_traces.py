"""Re-normalize ICD-10 codes in saved env traces and recompute their rewards.

Traces written before the env's submit path normalized diagnoses were scored on the raw
codes the policy emitted. This replays each episode's submit_diagnosis call through the
env's current normalization (invalid-code correction, name-based code resolution, acuity,
dedup) and re-scores it with the same scorer the env uses.

    python scripts/renormalize_traces.py results/traces/env_qwen3-4b_patient_diagnosis_heldout.jsonl
    python scripts/renormalize_traces.py results/traces/*.jsonl --in-place

Needs the benchmark database (DATABASE_URL, same as eval.config). `messages` are left
untouched: they record what the policy actually submitted. Updated per episode:
  reward, metrics              recomputed from the normalized submission
  meta.reward_original         the reward before this script (kept across reruns)
  meta.metrics_original        the metrics before this script (kept across reruns)
  ground_truth                 the instance's ground-truth diagnoses/codes, fetched from the
                               database when the trace predates recording them
  meta.submission_normalized   the problem list that was scored
  meta.icd10_corrected/resolved  codes fixed by prefix correction / from the name
  meta.icd10_resolved_snomed   the subset of resolved codes found through SNOMED CT
Without --in-place the result goes to <name>.renormalized.jsonl next to the input; with it
the input is replaced and the original kept as <name>.jsonl.bak. Non patient_diagnosis
episodes are copied through unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epic_sim.app.services.env_service import _code_fixes, normalize_submission  # noqa: E402
from eval.config import get_pg_connection  # noqa: E402
from eval.score_one import load_instance, score_submission  # noqa: E402

SUBMIT_TOOL = "submit_diagnosis"


def submitted_arguments(record: dict) -> dict | list | None:
    """Arguments of the episode's (last) submit_diagnosis call, or None if there is none."""
    args = None
    for msg in record.get("messages", []):
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") == SUBMIT_TOOL:
                args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args


def renormalize(record: dict, conn) -> dict:
    """Return `record` with the normalized submission's reward and metrics."""
    meta = record.get("meta", {})
    if meta.get("task") != "patient_diagnosis" or meta.get("gt_id") is None:
        return record
    prediction = normalize_submission("patient_diagnosis", submitted_arguments(record))
    result = score_submission(conn, meta["gt_id"], prediction or {}, "patient_diagnosis", None)
    ground_truth = record.get("ground_truth") or load_instance(conn, meta["gt_id"])["ground_truth"]
    return {
        **record,
        "ground_truth": ground_truth,
        "reward": result["reward"],
        "metrics": result["metrics"],
        "meta": {
            **meta,
            "reward_original": meta.get("reward_original", record.get("reward")),
            "metrics_original": meta.get("metrics_original", record.get("metrics")),
            "submission_normalized": prediction,
            **_code_fixes(prediction),
        },
    }


def process(path: Path, out: Path, conn) -> None:
    before, after, changed, regressed, corrected, resolved, via_snomed, n = 0.0, 0, 0.0, 0, 0, 0, 0, 0
    with open(path) as src, open(out, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            new = renormalize(record, conn)
            if new is not record:
                n += 1
                old_reward, new_reward = new["meta"]["reward_original"], new["reward"]
                before += old_reward or 0.0
                after += new_reward or 0.0
                changed += abs((new_reward or 0.0) - (old_reward or 0.0)) > 1e-9
                regressed += (new_reward or 0.0) < (old_reward or 0.0) - 1e-9
                corrected += new["meta"]["icd10_corrected"]
                resolved += new["meta"]["icd10_resolved"]
                via_snomed += new["meta"]["icd10_resolved_snomed"]
            dst.write(json.dumps(new, default=str) + "\n")
    if n:
        print(f"{path.name}: {n} episodes, reward changed on {changed} ({regressed} lower), mean {before / n:.4f} -> {after / n:.4f}, "
              f"codes corrected {corrected}, resolved from name {resolved} ({via_snomed} via SNOMED)")
    else:
        print(f"{path.name}: no patient_diagnosis episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+", type=Path, help="trace .jsonl files")
    ap.add_argument("--in-place", action="store_true", help="replace the input (original kept as .bak)")
    args = ap.parse_args()

    with get_pg_connection() as conn:
        for path in args.traces:
            out = path.with_suffix(".renormalized.jsonl") if not args.in_place else path.with_suffix(".jsonl.tmp")
            process(path, out, conn)
            if args.in_place:
                os.replace(path, path.with_suffix(".jsonl.bak"))
                os.replace(out, path)


if __name__ == "__main__":
    main()

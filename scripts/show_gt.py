"""Print the ground-truth diagnosis list for one or more gt_ids (patient_diagnosis task).

    python scripts/show_gt.py 7920 7820
    python scripts/show_gt.py 7920 --json

Reads benchmark_ground_truth through DATABASE_URL (eval.config.get_pg_connection). Entries
marked excluded_nondiagnostic are shown flagged; the scorer ignores them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import get_pg_connection  # noqa: E402
from eval.score_one import load_instance  # noqa: E402


def ground_truth_diagnoses(conn, gt_id: int) -> list[dict]:
    """[{list, icd10, name, acuity, excluded}] for an instance, active entries first."""
    gt = load_instance(conn, gt_id)["ground_truth"]
    rows = []
    for key, label in (("active_diagnoses", "active"), ("chronic_conditions", "chronic")):
        for d in gt.get(key, []):
            rows.append({"list": label, "icd10": d.get("icd10"), "name": d.get("display_name"),
                         "acuity": d.get("acuity"), "excluded": bool(d.get("excluded_nondiagnostic"))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_ids", type=int, nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with get_pg_connection() as conn:
        for g in args.gt_ids:
            inst = load_instance(conn, g)
            rows = ground_truth_diagnoses(conn, g)
            if args.json:
                print(json.dumps({"gt_id": g, "patient_id": inst["patient_id"], "split": inst["split"],
                                  "diagnoses": rows}))
                continue
            print(f"gt_id {g}  patient {inst['patient_id']}  split {inst['split']}  task {inst['task']}")
            for r in rows:
                print(f"  {r['list']:8s} {r['icd10'] or '':9s} {r['acuity'] or '':17s} {r['name']}"
                      + ("   [excluded]" if r["excluded"] else ""))


if __name__ == "__main__":
    main()

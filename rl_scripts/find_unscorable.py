"""List instances whose reward ceiling is below 1.0, to keep them out of the RL training pool.

For every instance of a split the script replays the label-derived oracle answer through the
env's submit path (the same normalization + scoring a rollout gets) and records the reward. An
instance whose oracle reward is under --threshold cannot be solved perfectly, so training on it
rewards the policy for gaming the normalizer. The cause is recorded per instance:

  dedup    the oracle has several codes in one 3-char category (5-char for S/T) and the
           submit-time dedup merges them, so the answer loses entries
  invalid  an oracle code is not in the loaded ICD-10-CM table, so the submit-time
           validator rewrites it to a parent code that no longer matches the label

    python scripts/find_unscorable.py --split train --out data/rl_exclude.json

The file is read by scripts/reward_variance.py (--exclude data/rl_exclude.json) and is meant for
the RL training loader too. Evaluation splits keep these instances; only the training pool drops them.
Restart the env server after changing RESOLVE_ICD10_FROM_NAME or the ICD table, then rerun this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.env_client import SyntheticHospitalEnv  # noqa: E402
from eval.icd10_validator import is_valid_icd10  # noqa: E402
from eval.tasks.diagnosis import normalize_icd10  # noqa: E402
from eval.tasks.patient_diagnosis import normalize_problem_list  # noqa: E402

TASK = "patient_diagnosis"


def diagnose(oracle: dict) -> list[str]:
    """Why a perfect answer may lose entries in the submit path (empty list: nothing found)."""
    dx = list(oracle.get("active_diagnoses", [])) + list(oracle.get("chronic_conditions", []))
    causes = []
    # resolve_from_name is deliberately off here: it mirrors RESOLVE_ICD10_FROM_NAME = False.
    active, chronic = normalize_problem_list(oracle.get("active_diagnoses"), oracle.get("chronic_conditions"),
                                             resolve_from_name=False)
    if len(active) + len(chronic) < len(dx):
        causes.append("dedup")
    if any(d.get("icd10") and not is_valid_icd10(normalize_icd10(d["icd10"])) for d in dx):
        causes.append("invalid")
    return causes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["public", "heldout", "train"])
    ap.add_argument("--threshold", type=float, default=0.999, help="oracle reward below this -> excluded")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=os.environ.get("SH_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production"))
    ap.add_argument("--out", default="data/rl_exclude.json")
    args = ap.parse_args()

    with SyntheticHospitalEnv(args.base_url, scorer_token=args.token) as env:
        items: list[dict] = []
        while True:
            chunk = env.instances(TASK, args.split, limit=500, offset=len(items))
            items.extend(chunk)
            if len(chunk) < 500:
                break
    print(f"{len(items)} instances in {TASK}/{args.split}", flush=True)

    local = threading.local()

    def check(item: dict) -> dict:
        if not hasattr(local, "env"):
            local.env = SyntheticHospitalEnv(args.base_url, scorer_token=args.token, timeout=300.0)
        env = local.env
        gt_id = item["gt_id"]
        try:
            obs = env.reset(gt_id=gt_id)
            oracle = env._client.get(f"/env/oracle/{env.episode_id}").json()["arguments"]
            _, reward, _, info = env.step(obs.submit_tool, oracle)
        except Exception as e:  # noqa: BLE001
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
            return {"gt_id": gt_id, "error": repr(e)[:200]}
        return {"gt_id": gt_id, "oracle_reward": reward, "causes": diagnose(oracle),
                "n_oracle_codes": len(oracle.get("active_diagnoses", [])) + len(oracle.get("chronic_conditions", []))}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(check, items))

    errors = [r for r in rows if "error" in r]
    scored = [r for r in rows if "error" not in r]
    bad = sorted((r for r in scored if r["oracle_reward"] < args.threshold), key=lambda r: r["oracle_reward"])
    unexplained = [r for r in bad if not r["causes"]]
    mean = sum(r["oracle_reward"] for r in scored) / max(len(scored), 1)
    print(f"oracle mean reward {mean:.4f} over {len(scored)} instances; "
          f"{len(bad)} below {args.threshold} ({len(bad) / max(len(scored), 1):.1%})")
    for cause in ("dedup", "invalid"):
        print(f"  cause {cause}: {sum(cause in r['causes'] for r in bad)}")
    print(f"  no cause identified: {len(unexplained)}"
          + (f"  <- inspect these by hand, e.g. gt_id {unexplained[0]['gt_id']}" if unexplained else ""))
    if errors:
        print(f"WARNING: {len(errors)} instances errored and are NOT in the exclusion list "
              f"(first: gt_id {errors[0]['gt_id']} {errors[0]['error']}); rerun before trusting it")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"task": TASK, "split": args.split, "threshold": args.threshold,
                   "n_scored": len(scored), "oracle_mean_reward": mean,
                   "gt_ids": sorted(r["gt_id"] for r in bad),
                   "details": {str(r["gt_id"]): {"oracle_reward": round(r["oracle_reward"], 4), "causes": r["causes"]}
                               for r in bad}}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

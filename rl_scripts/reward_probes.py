"""Probe the patient_diagnosis reward with model-free policies, to see what it pays for.

For each sampled instance the script reads the label-derived oracle answer (scorer-only
endpoint) and the chart's problem list, then submits a family of synthetic answers, each in
its own episode, through the same path a rollout uses (normalization + scoring):

  oracle                    sanity: should be ~1.0
  empty / malformed         should be 0
  chart_problem_list_names  copy the chart's history list verbatim (names only, codes resolved server-side)
  prior_top{3,5,10}         the K most frequent codes over *other* train instances, no chart reading
  oracle_keep_half          recall halved, precision intact
  oracle_plus_{5,20}_junk   oracle plus unrelated frequent codes: how hard is over-prediction punished?
  oracle_all_active / _all_chronic   acuity placement changed, codes identical
  oracle_codes_3char        codes cut to the 3-char category (does specificity matter?)
  oracle_names_only / _codes_only    drop the codes / drop the names
  oracle_x3_duplicates      every entry tripled (dedup check)

    python scripts/reward_probes.py --instances 100 --out results/reward_probes.jsonl

What to look for: any answer that needs no chart reading (prior_top*, junk, duplicates,
3-char codes) scoring close to the oracle is something RL will find and exploit. The prior
set is disjoint from the evaluated instances so prior_top* is an honest no-reading baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.env_client import SyntheticHospitalEnv  # noqa: E402

TASK = "patient_diagnosis"
METRIC_KEYS = ["problem_list_recall", "problem_list_precision", "problem_list_precision_neutral",
               "weighted_problem_list_recall", "acuity_accuracy"]


def flat(sub: dict) -> list[dict]:
    return list(sub.get("active_diagnoses", [])) + list(sub.get("chronic_conditions", []))


def entry(code: str, name: str, acuity: str = "acute") -> dict:
    return {"icd10": code, "name": name, "acuity": acuity}


def build_probes(ora: dict, chart: list[str], prior: list[tuple[str, str]]) -> dict[str, dict]:
    """prior: [(code, name)] most-frequent-first. Returns probe name -> submit arguments."""
    dx = flat(ora)
    cats = {d["icd10"][:3] for d in dx}
    junk = [(c, n) for c, n in prior if c[:3] not in cats]
    probes: dict[str, dict] = {
        "oracle": ora,
        "empty": {"active_diagnoses": [], "chronic_conditions": []},
        "malformed": {"note": "no diagnosis keys"},
        "chart_problem_list_names": {"active_diagnoses": [],
                                     "chronic_conditions": [entry("", n, "chronic") for n in chart]},
        "oracle_keep_half": {"active_diagnoses": ora["active_diagnoses"][::2],
                             "chronic_conditions": ora["chronic_conditions"][::2]},
        "oracle_all_active": {"active_diagnoses": [entry(d["icd10"], d["name"]) for d in dx],
                              "chronic_conditions": []},
        "oracle_all_chronic": {"active_diagnoses": [],
                               "chronic_conditions": [entry(d["icd10"], d["name"], "chronic") for d in dx]},
        "oracle_codes_3char": {k: [{**d, "icd10": d["icd10"][:3]} for d in ora[k]]
                               for k in ("active_diagnoses", "chronic_conditions")},
        "oracle_names_only": {k: [{**d, "icd10": ""} for d in ora[k]]
                              for k in ("active_diagnoses", "chronic_conditions")},
        "oracle_codes_only": {k: [{**d, "name": ""} for d in ora[k]]
                              for k in ("active_diagnoses", "chronic_conditions")},
        "oracle_x3_duplicates": {k: ora[k] * 3 for k in ("active_diagnoses", "chronic_conditions")},
    }
    for k in (3, 5, 10):
        probes[f"prior_top{k}"] = {"active_diagnoses": [entry(c, n) for c, n in prior[:k]],
                                   "chronic_conditions": []}
    for k in (5, 20):
        extra = [entry(c, n) for c, n in junk[:k]]
        probes[f"oracle_plus_{k}_junk"] = {"active_diagnoses": ora["active_diagnoses"] + extra,
                                           "chronic_conditions": ora["chronic_conditions"]}
    return copy.deepcopy(probes)


class Runner:
    def __init__(self, args):
        self.args = args
        self.local = threading.local()

    def env(self) -> SyntheticHospitalEnv:
        if not hasattr(self.local, "env"):
            self.local.env = SyntheticHospitalEnv(self.args.base_url, scorer_token=self.args.token, timeout=120.0)
        return self.local.env

    def oracle(self, gt_id: int, with_chart: bool = False) -> tuple[dict, list[str]]:
        """One throwaway episode: read the oracle (and optionally the chart's problem list)."""
        env = self.env()
        obs = env.reset(gt_id=gt_id)
        try:
            ora = env._client.get(f"/env/oracle/{env.episode_id}").json()["arguments"]
            chart: list[str] = []
            if with_chart:
                result = env.step("view_problem_list", {"patient_id": obs.patient_id})[0]
                chart = [p.get("display_name", "") for p in result if isinstance(p, dict)] if isinstance(result, list) else []
        finally:
            env.close()
        return ora, chart

    def score(self, gt_id: int, probe_args: dict) -> tuple[float, dict]:
        env = self.env()
        obs = env.reset(gt_id=gt_id)
        _, reward, done, info = env.step(obs.submit_tool, probe_args)
        assert done
        return reward, info.get("metrics") or {}

    def run_instance(self, gt_id: int, prior: list[tuple[str, str]]) -> list[dict]:
        ora, chart = self.oracle(gt_id, with_chart=True)
        rows = []
        for name, sub in build_probes(ora, chart, prior).items():
            try:
                reward, metrics = self.score(gt_id, sub)
            except Exception as e:  # noqa: BLE001
                print(f"  gt_id={gt_id} probe={name}: {e!r}", flush=True)
                continue
            rows.append({"gt_id": gt_id, "probe": name, "reward": reward,
                         "n_pred": len(flat(sub)), "n_gt": len(flat(ora)),
                         **{k: metrics.get(k) for k in METRIC_KEYS}})
        return rows


def prior_codes(runner: Runner, gt_ids: list[int], workers: int) -> list[tuple[str, str]]:
    def one(g):
        try:
            return flat(runner.oracle(g)[0])
        except Exception:  # noqa: BLE001
            return []
    with ThreadPoolExecutor(workers) as pool:
        counts: Counter = Counter()
        names: dict[str, str] = {}
        for dx in pool.map(one, gt_ids):
            for d in dx:
                counts[d["icd10"]] += 1
                names.setdefault(d["icd10"], d.get("name", ""))
    return [(c, names[c]) for c, _ in counts.most_common(50)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["public", "heldout", "train"])
    ap.add_argument("--instances", type=int, default=100, help="instances to probe")
    ap.add_argument("--prior-instances", type=int, default=200,
                    help="further, disjoint instances used to build the frequency prior")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=os.environ.get("SH_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production"))
    ap.add_argument("--out", default=None, help="write one JSON line per (instance, probe)")
    args = ap.parse_args()

    runner = Runner(args)
    with SyntheticHospitalEnv(args.base_url, scorer_token=args.token) as env:
        items: list[dict] = []
        while True:
            chunk = env.instances(TASK, args.split, limit=500, offset=len(items))
            items.extend(chunk)
            if len(chunk) < 500:
                break
    ids = [it["gt_id"] for it in items]
    random.Random(args.seed).shuffle(ids)
    eval_ids = ids[:args.instances]
    prior_ids = ids[args.instances:args.instances + args.prior_instances]
    print(f"{len(ids)} instances; probing {len(eval_ids)}, prior from {len(prior_ids)} others")
    if len(prior_ids) < 20:
        print("WARNING: too few instances left over for a meaningful frequency prior")

    prior = prior_codes(runner, prior_ids, args.workers)
    print("prior (top 10 codes): " + ", ".join(f"{c} ({n[:24]})" for c, n in prior[:10]))

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for batch in pool.map(lambda g: runner.run_instance(g, prior), eval_ids) for r in batch]
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    report(rows)


def report(rows: list[dict]) -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["probe"]].append(r)
    base = {r["gt_id"]: r["reward"] for r in by.get("oracle", [])}

    def m(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'probe':28s} {'n':>4s} {'mean':>6s} {'median':>6s} {'>=.99':>6s} {'==0':>6s} "
          f"{'recall':>6s} {'prec':>6s} {'prec_n':>6s} {'acuity':>6s}")
    order = list(by)
    for name in order:
        rs = by[name]
        rw = np.array([r["reward"] for r in rs])
        print(f"{name:28s} {len(rs):4d} {rw.mean():6.3f} {np.median(rw):6.3f} {(rw >= .99).mean():6.2f} "
              f"{(rw == 0).mean():6.2f} {m(rs, 'problem_list_recall'):6.2f} {m(rs, 'problem_list_precision'):6.2f} "
              f"{m(rs, 'problem_list_precision_neutral'):6.2f} {m(rs, 'acuity_accuracy'):6.2f}")

    mean = {k: float(np.mean([r["reward"] for r in v])) for k, v in by.items()}
    o = mean.get("oracle", float("nan"))
    print(f"\nnotes (oracle mean {o:.3f}):")
    low = sorted((v, g) for g, v in base.items() if v < 0.95)
    if low:
        print(f"  - oracle scores <0.95 on {len(low)}/{len(base)} instances (lowest: "
              + ", ".join(f"gt_id {g}={v:.2f}" for v, g in low[:5])
              + "): the reward ceiling is below 1 there; check whether that is expected")
    for name in ("empty", "malformed"):
        if mean.get(name, 0) > 0:
            print(f"  - {name} submission earns {mean[name]:.3f}, expected 0")
    for k in (3, 5, 10):
        v = mean.get(f"prior_top{k}")
        if v is not None:
            print(f"  - no-chart prior (top {k} frequent codes) earns {v:.3f} = {v / o:.0%} of oracle")
    for k in (5, 20):
        v = mean.get(f"oracle_plus_{k}_junk")
        if v is not None:
            print(f"  - oracle + {k} junk codes earns {v:.3f} ({v - o:+.3f} vs oracle): "
                  + ("over-prediction is barely punished" if o - v < 0.1 * o else "over-prediction is punished"))
    for name, what in [("oracle_codes_3char", "full-specificity codes"), ("oracle_x3_duplicates", "dedup"),
                       ("oracle_all_active", "acuity placement (all active)"),
                       ("oracle_all_chronic", "acuity placement (all chronic)"),
                       ("oracle_names_only", "explicit ICD-10 codes"), ("oracle_codes_only", "diagnosis names")]:
        v = mean.get(name)
        if v is not None:
            print(f"  - {name}: {v:.3f} ({v - o:+.3f}): " + ("reward is insensitive to " + what
                  if abs(v - o) < 0.02 else "reward changes with " + what))
    v = mean.get("oracle_codes_only")
    if v is not None and v - o > 0.01:
        print(f"  - oracle answer WITHOUT names beats the oracle by {v - o:+.3f}: submit-time name-based code")
    v = mean.get("chart_problem_list_names")
    if v is not None:
        print(f"  - copying the chart's history list earns {v:.3f} = {v / o:.0%} of oracle")


if __name__ == "__main__":
    main()

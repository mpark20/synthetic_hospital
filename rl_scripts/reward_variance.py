"""Sample K episodes per instance and measure whether the reward gives RL a signal.

GRPO-style RL learns from the *difference* between rollouts of the same prompt. If every
sample of an instance scores the same (all 0, or all ~1), that prompt contributes no gradient.
This script rolls the policy out K times on N train instances with env_run.run_episode (the
same loop used for evaluation), writes one JSON line per rollout, then reports:

  - reward distribution, and how often the policy fails to submit / runs out of budget
  - within-instance spread: share of zero-variance groups (all-zero / all-high / constant-mid),
    share of "informative" groups (reward range >= --min-range), and the split of total reward
    variance into within-instance (what sampling can exploit) vs between-instance (difficulty)
  - best-of-K vs mean-of-K (the headroom RL can try to close)
  - whether more tool use helps *within* an instance, and a breakdown by instance difficulty
  - trajectory length (to fix the training context and budget): steps, the first prompt, peak
    context per episode (the largest prompt + completion of any turn, NOT the sum over turns),
    tokens generated per turn and per episode, how much of each tool's output is cut off by the
    server's 8,000-character limit, and which context windows would hold 90/95/99% of episodes

# policy served by vLLM (registry entry qwen3-4b -> http://localhost:30001/v1)
python scripts/reward_variance.py --model qwen3-4b --instances 100 --samples 8 \
    --temperature 1.0 --workers 16 --exclude data/rl_exclude.json \
    --out results/reward_variance_qwen3-4b.jsonl

# re-read finished results without sampling
python scripts/reward_variance.py --analyze-only results/reward_variance_qwen3-4b.jsonl \
    --exclude data/rl_exclude.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))    # for env_run (scripts/ is not a package)

from env_run import run_episode, split_instances  # noqa: E402
from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402
from eval.env_client import SyntheticHospitalEnv  # noqa: E402

TASK = "patient_diagnosis"
METRIC_KEYS = ["problem_list_recall", "problem_list_precision_neutral", "weighted_problem_list_recall",
               "weighted_problem_list_f1_neutral", "acuity_accuracy"]
TRUNCATION_MARK = "[OUTPUT TRUNCATED]"      # env_service.render appends this past MAX_OBS_CHARS
CONTEXT_WINDOWS = [8192, 12288, 16384, 24576, 32768, 49152, 65536]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TracingAdapter:
    """Wraps an adapter and records the token usage of every model call in one rollout.

    run_episode only reports totals summed over turns; the context a trainer must hold is
    the largest single prompt (+ its completion), which needs the per-call numbers.
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self.calls: list[tuple[int, int]] = []      # (prompt_tokens, completion_tokens) per call

    def call_multi_turn_with_retry(self, *args, **kwargs):
        resp = self._adapter.call_multi_turn_with_retry(*args, **kwargs)
        self.calls.append((resp.input_tokens, resp.output_tokens))
        return resp


def trajectory_stats(calls: list[tuple[int, int]], transcript: list[dict]) -> dict:
    """Per-rollout length numbers from the per-call usage and the SFT-style transcript."""
    prompt = [c[0] for c in calls]
    comp = [c[1] for c in calls]
    names: dict[str, str] = {}
    for m in transcript:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
    tools = [{"tool": names.get(m.get("tool_call_id"), "?"), "chars": len(m["content"]),
              "truncated": TRUNCATION_MARK in m["content"]}
             for m in transcript if m.get("role") == "tool"]
    return {
        "n_calls": len(calls),
        "first_prompt_tokens": prompt[0] if prompt else 0,
        "peak_context": max((p + c for p, c in calls), default=0),
        "max_turn_output_tokens": max(comp, default=0),
        "turn_output_tokens": comp,
        "tool_obs": tools,
    }


def run_sample(env: SyntheticHospitalEnv, adapter, item: dict, sample: int, budget: int | None,
               keep_transcript: bool, max_tokens: int | None = None) -> dict:
    """One rollout -> one compact record. Failures become records with `error` set."""
    rec = {"gt_id": item["gt_id"], "sample": sample, "difficulty": item.get("difficulty"),
           "patient_id": item.get("patient_id"), "max_tokens": max_tokens}
    t0 = time.monotonic()
    traced = TracingAdapter(adapter)
    try:
        obs = env.reset(gt_id=item["gt_id"], budget=budget)
        res = run_episode(env, traced, obs)
    except Exception as e:  # noqa: BLE001: model/API/server failure; not a zero-reward sample
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        return rec | {"error": repr(e)[:300], "seconds": round(time.monotonic() - t0, 1)}
    metrics = res.get("metrics") or {}
    rec |= {
        "reward": res["reward"], "done": res["done"], "steps": res["steps"], "budget": obs.budget,
        "nudges": res["nudges"], "malformed": res["malformed_tool_calls"],
        "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],   # sums over turns
        "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        "seconds": round(time.monotonic() - t0, 1),
        **trajectory_stats(traced.calls, res["transcript"]),
    }
    if keep_transcript:
        rec["transcript"] = res["transcript"]
    if not res["done"]:
        try:
            env.close()     # ran out of turns without submitting
        except Exception:  # noqa: BLE001
            pass
    return rec


def finished(path: str) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if os.path.exists(path):
        for r in load(path):
            if not r.get("error"):
                done.add((r["gt_id"], r["sample"]))
    return done


def load(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def load_exclude(path: str | None) -> set[int]:
    """gt_ids to leave out (the file written by scripts/find_unscorable.py)."""
    if not path:
        return set()
    with open(path) as f:
        return set(json.load(f)["gt_ids"])


def sample_all(args) -> None:
    cfg = MODEL_REGISTRY[args.model]
    cfg = dataclasses.replace(cfg, temperature=args.temperature,
                              base_url=args.model_url or cfg.base_url,
                              model_id=args.model_id or cfg.model_id)
    os.environ.setdefault("EVAL_TRACE", "0")

    with SyntheticHospitalEnv(args.base_url, scorer_token=args.token) as env:
        items = split_instances(env, TASK, args.split)
    exclude = load_exclude(args.exclude)
    n_all = len(items)
    items = [it for it in items if it["gt_id"] not in exclude]
    if exclude:
        print(f"excluded {n_all - len(items)} of {n_all} instances using {args.exclude}", flush=True)
    random.Random(args.seed).shuffle(items)
    items = items[:args.instances]
    done = finished(args.out)
    todo = [(it, s) for it in items for s in range(args.samples) if (it["gt_id"], s) not in done]
    print(f"{len(items)} instances x {args.samples} samples; {len(done)} already in {args.out}, "
          f"running {len(todo)} at temperature {cfg.temperature} against {cfg.base_url} ({cfg.model_id})",
          flush=True)

    local = threading.local()
    lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    out = open(args.out, "a")

    def work(job):
        item, s = job
        if not hasattr(local, "env"):       # one env client and one adapter per thread
            local.env = SyntheticHospitalEnv(args.base_url, scorer_token=args.token)
            local.adapter = create_adapter(cfg)
        rec = run_sample(local.env, local.adapter, item, s, args.budget, args.keep_transcripts,
                         max_tokens=cfg.max_tokens)
        with lock:
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            counter["n"] += 1
            counter["err"] += bool(rec.get("error"))
            print(f"[{counter['n']}/{len(todo)}] gt_id={rec['gt_id']} s={s} "
                  + (f"ERROR {rec['error']}" if rec.get("error") else
                     f"reward={rec['reward']:.3f} steps={rec['steps']} done={rec['done']}"), flush=True)
            if counter["err"] > args.max_errors:
                print("too many errors; inspect the model and env servers", flush=True)
                os._exit(1)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    out.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def hist(rewards: np.ndarray, width: int = 40) -> None:
    edges = np.linspace(0, 1, 11)
    zero = int((rewards == 0).sum())
    counts = np.histogram(rewards[rewards > 0], bins=edges)[0]
    rows = [("0", zero)] + [(f"({edges[i]:.1f},{edges[i + 1]:.1f}]", int(c)) for i, c in enumerate(counts)]
    peak = max(c for _, c in rows) or 1
    for label, c in rows:
        print(f"  {label:>10s} {c:5d} {rewards.size and c / rewards.size:6.1%} " + "#" * round(width * c / peak))


def pcts(x, qs=(50, 90, 95, 99)) -> str:
    a = np.asarray(x, dtype=float)
    return "  ".join(f"p{q} {np.percentile(a, q):,.0f}" for q in qs) + f"  max {a.max():,.0f}"


def trajectory_report(ok: list[dict]) -> None:
    """Episode length numbers that fix the training context window and action budget."""
    t = [r for r in ok if "peak_context" in r]
    print("\nTRAJECTORY LENGTH")
    if not t:
        print("  no per-turn data in these records (they predate trajectory tracking); resample to get it")
        return
    steps = [r["steps"] for r in ok]
    budget = ok[0]["budget"]
    print(f"  steps          {pcts(steps)}   (budget {budget}; {np.mean([s >= budget for s in steps]):.1%} spend all of it)")
    first = [r["first_prompt_tokens"] for r in t]
    peak = np.array([r["peak_context"] for r in t])
    print(f"  first prompt   mean {np.mean(first):,.0f} tokens (system prompt + tool schemas + assignment, before any step)")
    print(f"  peak context   {pcts(peak)}   (largest prompt + completion of any turn)")
    per_step = [(r["peak_context"] - r["first_prompt_tokens"]) / max(r["n_calls"] - 1, 1) for r in t if r["n_calls"] > 1]
    if per_step:
        print(f"  context growth median {np.median(per_step):,.0f} tokens per turn (mean {np.mean(per_step):,.0f})")
    print("  episodes that fit in a context window: "
          + "   ".join(f"{w // 1024}k {np.mean(peak <= w):.0%}" for w in CONTEXT_WINDOWS))
    fit = next((w for w in CONTEXT_WINDOWS if np.mean(peak <= w) >= 0.95), None)
    print(f"  -> smallest listed window holding >=95% of episodes: "
          + (f"{fit // 1024}k" if fit else f"none up to {CONTEXT_WINDOWS[-1] // 1024}k; shrink observations or the budget"))

    turns = [x for r in t for x in r["turn_output_tokens"]]
    per_ep = [sum(r["turn_output_tokens"]) for r in t]
    cap = t[0].get("max_tokens")
    hit = np.mean([x >= cap for x in turns]) if cap else float("nan")
    print(f"  completion tokens per turn    {pcts(turns)}   (generation cap {cap}; turns that hit it: {hit:.1%})")
    print(f"  completion tokens per episode {pcts(per_ep)}   (the tokens the loss is computed on)")

    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in t:
        for o in r["tool_obs"]:
            by_tool[o["tool"]].append(o)
    print(f"  tool outputs (server cuts at 8,000 chars):")
    print(f"    {'tool':24s} {'calls':>6s} {'mean chars':>10s} {'p95 chars':>10s} {'truncated':>9s}")
    for tool, obs in sorted(by_tool.items(), key=lambda kv: -sum(o["chars"] for o in kv[1])):
        ch = [o["chars"] for o in obs]
        print(f"    {tool:24s} {len(obs):6d} {np.mean(ch):10,.0f} {np.percentile(ch, 95):10,.0f} "
              f"{np.mean([o['truncated'] for o in obs]):9.1%}")

    # do long episodes do worse? (quartiles of peak context)
    rw = np.array([r["reward"] for r in t])
    if len(t) >= 8 and np.ptp(peak) > 0:
        edges = np.percentile(peak, [0, 25, 50, 75, 100])
        print("  reward by peak-context quartile: " + "   ".join(
            f"<= {edges[i + 1]:,.0f}: {rw[(peak >= edges[i]) & (peak <= edges[i + 1])].mean():.3f}" for i in range(4)))
    if np.mean([s >= budget for s in steps]) < 0.05 and np.percentile(steps, 95) < 0.6 * budget:
        print(f"  hint: 95% of episodes finish within {np.percentile(steps, 95):.0f} steps but the budget is {budget}; "
              f"a smaller budget would cap the long tail without cutting normal episodes (check the reward of "
              f"episodes above that step count first)")
    if cap and hit > 0.02:
        print("  hint: >2% of turns hit the generation cap, so some answers are cut off; raise max_tokens or fix verbosity")


def analyze(recs: list[dict], min_range: float, prompts_per_step: int) -> None:
    ok = [r for r in recs if not r.get("error")]
    print(f"{len(recs)} rollouts, {len(recs) - len(ok)} errored (excluded below)")
    if not ok:
        return
    rw = np.array([r["reward"] for r in ok])
    print(f"\nREWARD  mean {rw.mean():.3f}  median {np.median(rw):.3f}  std {rw.std():.3f}  "
          f"==0: {(rw == 0).mean():.1%}  >=0.99: {(rw >= 0.99).mean():.1%}")
    hist(rw)

    print("\nBEHAVIOUR")
    done = np.array([bool(r["done"]) for r in ok])
    steps = np.array([r["steps"] for r in ok])
    exhausted = np.array([r["steps"] >= r["budget"] for r in ok])
    print(f"  submitted before running out of turns: {done.mean():.1%}   hit the action budget: {exhausted.mean():.1%}")
    print(f"  steps mean {steps.mean():.1f} median {np.median(steps):.0f} max {steps.max()}   "
          f"nudges/episode {np.mean([r['nudges'] for r in ok]):.2f}   "
          f"malformed tool calls: {np.mean([bool(r['malformed']) for r in ok]):.1%}")
    print(f"  tokens per episode summed over turns (a cost measure, NOT context length; see TRAJECTORY LENGTH): "
          f"in {np.mean([r['input_tokens'] for r in ok]):,.0f} / out {np.mean([r['output_tokens'] for r in ok]):,.0f}   "
          f"{np.mean([r['seconds'] for r in ok]):.0f}s per episode")
    print(f"  reward when submitted: {rw[done].mean() if done.any() else float('nan'):.3f}   "
          f"when not: {rw[~done].mean() if (~done).any() else float('nan'):.3f}")

    print("\nMETRICS (mean over rollouts)")
    for k in METRIC_KEYS:
        v = [r["metrics"][k] for r in ok if r.get("metrics", {}).get(k) is not None]
        if v:
            print(f"  {k:36s} {np.mean(v):.3f}")

    trajectory_report(ok)

    groups: dict[int, list[dict]] = defaultdict(list)
    for r in ok:
        groups[r["gt_id"]].append(r)
    groups = {g: v for g, v in groups.items() if len(v) >= 2}
    if not groups:
        print("\nno instance has >=2 valid samples; cannot compute within-instance statistics")
        return
    ks = [len(v) for v in groups.values()]
    gr = {g: np.array([r["reward"] for r in v]) for g, v in groups.items()}
    means = np.array([a.mean() for a in gr.values()])
    stds = np.array([a.std() for a in gr.values()])
    rng_ = np.array([a.max() - a.min() for a in gr.values()])
    zero_var = stds < 1e-6
    n = len(gr)
    all_zero = zero_var & (means == 0)
    all_high = zero_var & (means >= 0.99)
    informative = rng_ >= min_range
    print(f"\nWITHIN-INSTANCE SIGNAL  ({n} instances, K = {min(ks)}..{max(ks)} samples each)")
    print(f"  zero-variance groups: {zero_var.mean():.1%}  "
          f"(all-zero {all_zero.mean():.1%}, all-high {all_high.mean():.1%}, "
          f"constant-mid {(zero_var & ~all_zero & ~all_high).mean():.1%})")
    print(f"  informative groups (reward range >= {min_range}): {informative.mean():.1%}  "
          f"-> ~{informative.mean() * prompts_per_step:.0f} of {prompts_per_step} prompts per step carry gradient")
    print(f"  mean within-group std {stds.mean():.3f}   median range {np.median(rng_):.3f}")
    within = float(np.mean([a.var() for a in gr.values()]))
    between = float(means.var())
    print(f"  reward variance: within-instance {within:.4f}, between-instance {between:.4f} "
          f"({within / (within + between):.0%} within)")
    best = float(np.mean([a.max() for a in gr.values()]))
    print(f"  mean-of-K {means.mean():.3f}  best-of-K {best:.3f}  (headroom {best - means.mean():+.3f})")

    # does using more tools help *inside* an instance?
    dr, ds = [], []
    for v in groups.values():
        r_ = np.array([x["reward"] for x in v])
        s_ = np.array([x["steps"] for x in v], dtype=float)
        dr += list(r_ - r_.mean())
        ds += list(s_ - s_.mean())
    if np.std(ds) > 0 and np.std(dr) > 0:
        print(f"  within-instance corr(reward, steps) = {np.corrcoef(dr, ds)[0, 1]:+.2f}")

    print("\nBY DIFFICULTY")
    by_diff: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(groups.values()):
        by_diff[str(v[0].get("difficulty"))].append(i)
    for d, idx in sorted(by_diff.items()):
        print(f"  {d:10s} n={len(idx):4d}  mean {means[idx].mean():.3f}  informative {informative[idx].mean():.1%}  "
              f"all-zero {all_zero[idx].mean():.1%}  all-high {all_high[idx].mean():.1%}")

    print("\nHINTS (heuristic)")
    if informative.mean() < 0.3:
        print("  - few informative groups: RL will be sample-inefficient; raise K, raise temperature, or filter prompts")
    if all_zero.mean() > 0.25:
        print("  - many all-zero groups: look at those transcripts (--keep-transcripts); format/tool-use failures "
              "may call for a short SFT warm start or an easier prompt curriculum")
    if all_high.mean() > 0.25:
        print("  - many all-high groups: those prompts are solved; drop them from the training pool")
    if (~done).mean() > 0.05:
        print("  - >5% of rollouts never submit: raise max tokens/turns or fix the failure before training")
    if informative.mean() >= 0.3 and all_zero.mean() <= 0.25 and (~done).mean() <= 0.05:
        print("  - signal looks adequate for GRPO at this K and temperature")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze-only", metavar="JSONL", default=None, help="skip sampling; analyze this file")
    ap.add_argument("--model", default="qwen3-4b", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--model-url", default=None, help="override the registry entry's OpenAI-compatible base_url")
    ap.add_argument("--model-id", default=None, help="override the served model name")
    ap.add_argument("--split", default="train", choices=["public", "heldout", "train"])
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--samples", type=int, default=8, help="K rollouts per instance (your GRPO group size)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=None, help="action budget (server default 40)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent rollouts")
    ap.add_argument("--max-errors", type=int, default=25)
    ap.add_argument("--min-range", type=float, default=0.1, help="reward range that makes a group informative")
    ap.add_argument("--prompts-per-step", type=int, default=32, help="for the per-step estimate only")
    ap.add_argument("--exclude", default=None, metavar="JSON",
                    help="gt_ids to skip (scripts/find_unscorable.py output); also applied by --analyze-only")
    ap.add_argument("--keep-transcripts", action="store_true", help="store full transcripts (large)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=os.environ.get("SH_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production"))
    ap.add_argument("--out", default=None, help="results JSONL (default results/reward_variance_<model>.jsonl)")
    args = ap.parse_args()

    exclude = load_exclude(args.exclude)
    if args.analyze_only:
        analyze([r for r in load(args.analyze_only) if r["gt_id"] not in exclude], args.min_range, args.prompts_per_step)
        return
    args.out = args.out or f"results/reward_variance_{args.model}.jsonl"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample_all(args)
    analyze([r for r in load(args.out) if r["gt_id"] not in exclude], args.min_range, args.prompts_per_step)


if __name__ == "__main__":
    main()

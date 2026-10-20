"""Baseline an LLM on SyntheticHospitalEnv: run N episodes, report mean reward.

For each episode the model gets the env's system prompt, intro and tool schemas,
calls tools through env.step(), and the submit tool ends the episode with the
reward from the scoring endpoint. Models come from eval.config.MODEL_REGISTRY and
are called through eval.adapters (the same path eval/agents/runner.py uses).

    python scripts/env_baseline.py --model gpt-5.3 --task patient_diagnosis \
        --split public --episodes 20 --out results/env_baseline.jsonl

Episodes run sequentially (one model call in flight at a time).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402
from eval.env_client import SyntheticHospitalEnv  # noqa: E402

TASKS = ["patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication"]


def run_episode(env: SyntheticHospitalEnv, adapter, obs) -> dict:
    messages = [{"role": "user", "content": obs.intro}]
    reward, done, info = 0.0, False, {}
    steps, in_tok, out_tok = 0, 0, 0
    # Budget is enforced server-side (only the submit tool is accepted once spent);
    # the extra turns just let a model that ignores the warning still submit.
    for _ in range(obs.budget + 5):
        resp = adapter.call_multi_turn_with_retry(obs.instructions, messages, tools=obs.tools)
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens

        if not resp.tool_calls:
            messages.append({"role": "assistant", "content": resp.text or "(no action)"})
            messages.append({"role": "user", "content":
                             "Please call a tool or submit your answer with "
                             f"{obs.submit_tool}."})
            continue

        raw = resp.raw_tool_calls or []
        messages.append({"role": "assistant", "content": resp.text or "", "tool_calls": raw})
        for tc, raw_tc in zip(resp.tool_calls, raw):
            try:
                _, reward, done, info = env.step(tc["name"], tc["arguments"])
                text = info["observation_text"]
                steps = info["step"]
            except RuntimeError as e:       # e.g. episode already ended server-side
                text, done = f"error: {e}", True
            messages.append({"role": "tool", "tool_call_id": raw_tc.get("id", ""), "content": text})
            if done:
                break
        if done:
            break
    return {"reward": reward, "done": done, "steps": steps,
            "metric": info.get("reward_metric"), "metrics": info.get("metrics"),
            "input_tokens": in_tok, "output_tokens": out_tok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--task", default="patient_diagnosis", choices=TASKS)
    ap.add_argument("--split", default="public", choices=["public", "heldout", "train"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=None, help="action budget (server default 40)")
    ap.add_argument("--base-url", default=os.environ.get("SH_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production"))
    ap.add_argument("--out", default=None, help="append one JSON line per episode")
    args = ap.parse_args()

    adapter = create_adapter(MODEL_REGISTRY[args.model])
    rewards: list[float] = []
    metric_sums: dict[str, float] = defaultdict(float)
    out = open(args.out, "a") if args.out else None

    with SyntheticHospitalEnv(args.base_url, scorer_token=args.token, budget=args.budget) as env:
        for i in range(args.episodes):
            obs = env.reset(task=args.task, split=args.split, seed=args.seed + i)
            t0 = time.monotonic()
            try:
                res = run_episode(env, adapter, obs)
            except Exception as e:  # model/API failure: count as 0 rather than drop the instance
                env.close()
                res = {"reward": 0.0, "done": False, "steps": 0, "error": repr(e)}
            res.update(episode=i, gt_id=obs.gt_id, patient_id=obs.patient_id, task=args.task,
                       split=args.split, model=args.model, seconds=round(time.monotonic() - t0, 1))
            rewards.append(res["reward"])
            for k, v in (res.get("metrics") or {}).items():
                if isinstance(v, (int, float)):
                    metric_sums[k] += v
            print(f"[{i + 1}/{args.episodes}] gt_id={obs.gt_id} steps={res['steps']} "
                  f"reward={res['reward']:.3f} {res.get('error', '')}", flush=True)
            if out:
                out.write(json.dumps(res, default=str) + "\n")
                out.flush()

    n = max(len(rewards), 1)
    print(f"\n{args.model} | {args.task} | {args.split} | n={len(rewards)}")
    print(f"mean reward: {sum(rewards) / n:.4f}  (metric: {res.get('metric')})")
    print("mean metrics: " + ", ".join(f"{k}={v / n:.3f}" for k, v in sorted(metric_sums.items())))


if __name__ == "__main__":
    main()

"""Full-fidelity trace logging: the final LLM request/response per task as one JSONL line.

The DB keeps only a truncated `raw_output`, so prompts, thinking blocks and tool
calls are lost. This module records, per API call, the exact request payload
(system prompt, messages, tools) and the untouched response body (text, thinking
blocks, tool_use blocks, usage) to results/traces/run_<run_id>.jsonl.

Context (run_id, gt_id) is carried in contextvars, so set it inside the worker
thread that makes the calls (see `trace_context`). Calls made with no context are
written to results/traces/run_unassigned.jsonl.

Disable with EVAL_TRACE=0; relocate with EVAL_TRACE_DIR.
"""

import contextvars
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("eval_trace_ctx", default={})
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("EVAL_TRACE", "1") != "0"


def _trace_dir() -> str:
    return os.environ.get("EVAL_TRACE_DIR", "results/traces")


@contextmanager
def trace_context(**fields):
    """Tag all calls in this block, e.g. trace_context(run_id=3, gt_id=17)."""
    outer = _ctx.get()
    state = {"count": 0, "entry": None}
    token = _ctx.set({**outer, **fields, "_state": state})
    try:
        yield
    finally:
        _ctx.reset(token)
        if state["entry"] is not None:
            state["entry"]["call_index"] = state["count"]
            _write(state["entry"])


def record_call(*, adapter: str, model_id: str, request: dict, response_body,
                text: str, input_tokens: int, output_tokens: int,
                latency_ms: int, tool_calls: list[dict] | None = None,
                error: str | None = None) -> None:
    """Append one request/response pair. Never raises into the eval path."""
    if not enabled():
        return
    try:
        ctx = dict(_ctx.get())
        state = ctx.pop("_state", None)
        if isinstance(response_body, str):
            try:
                response_body = json.loads(response_body, strict=False)
            except ValueError:
                pass  # keep the raw string
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **ctx,
            "call_index": 1,
            "adapter": adapter,
            "model_id": model_id,
            "request": request,
            "response": response_body,
            "text": text,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            **({"error": error} if error else {}),
        }
        if state is None:
            _write(entry)  # no context to wait on
        else:
            state["count"] += 1
            state["entry"] = entry  # superseded by any later call
    except Exception:  # tracing must never break an evaluation
        import logging
        logging.getLogger(__name__).exception("trace logging failed")


def _write(entry: dict) -> None:
    try:
        run_id = entry.get("run_id")
        with _lock:
            os.makedirs(_trace_dir(), exist_ok=True)
            path = os.path.join(
                _trace_dir(), f"run_{run_id}.jsonl" if run_id is not None else "run_unassigned.jsonl"
            )
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        import logging
        logging.getLogger(__name__).exception("trace logging failed")

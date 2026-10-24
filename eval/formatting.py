"""Formats episode records for multi-turn env rollouts.

Each finished episode becomes one JSONL line in OpenAI chat format, loadable by
TRL / axolotl / torchtune without conversion:

    {"messages": [...], "tools": [...], "reward": 0.8, "metrics": {...}, "meta": {...}}

Only turns the policy produced are assistant messages; harness nudges ("please call a
tool") and failed turns are left out, so the transcript is what a trained model should
imitate. Assistant `tool_calls[].function.arguments` are dicts (what HF chat templates
expect); `reasoning_content` is kept as a separate field when the API returned thinking,
so it can be trained on or stripped at load time. Loss masking is left to the trainer
(e.g. TRL `assistant_only_loss`).

Only OpenAI-compatible responses are normalized so far (`assistant_message`).

"""

import json
import logging
import os
import threading

_lock = threading.Lock()
log = logging.getLogger(__name__)


class MalformedToolCall(ValueError):
    """A tool call whose arguments aren't a JSON object; such episodes aren't SFT-safe."""


def assistant_message(text: str, raw_tool_calls: list[dict], reasoning: str | None = None) -> dict:
    """OpenAI-compatible assistant turn -> canonical message (arguments as dicts)."""
    calls = []
    for tc in raw_tool_calls:
        fn = tc["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                raise MalformedToolCall(f"{fn['name']}: {e}") from e
        if not isinstance(args, dict):
            raise MalformedToolCall(f"{fn['name']}: arguments are {type(args).__name__}, not an object")
        calls.append({"id": tc.get("id", ""), "type": "function",
                      "function": {"name": fn["name"], "arguments": args}})
    msg = {"role": "assistant", "content": text or "", "tool_calls": calls}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def tool_message(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def write_episode(path: str, *, messages: list[dict], tools: list[dict], reward: float,
                  metrics: dict | None, meta: dict) -> None:
    """Append one episode to the JSONL file at `path`. Never raises into the eval path."""
    try:
        entry = {"messages": messages, "tools": tools, "reward": reward,
                 "metrics": metrics or {}, "meta": meta}
        with _lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        log.exception("episode formatting failed")

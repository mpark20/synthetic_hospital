#!/usr/bin/env python3
"""Convert EHR-agent run records (JSON or JSONL) into readable, self-contained HTML.

    python ehr_run_to_html.py run.json [out.html]
    python ehr_run_to_html.py runs.jsonl [out_dir]      # one page per line

    from ehr_run_to_html import render_record
    html_text = render_record(json.load(open("run.json")))
"""
from multiprocessing import Value
import html, json, re, sys
from collections import Counter
from pathlib import Path

esc = lambda s: html.escape("" if s is None else str(s), quote=True)
pretty = lambda o: json.dumps(o, indent=2, ensure_ascii=False)


def maybe_json(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) or "none"
    return str(v)


def details(summary, body, open_=False):
    return f'<details class="t"{" open" if open_ else ""}><summary>{summary}</summary>{body}</details>'


def flag(text, kind=""):
    return f'<span class="flag {kind}">{esc(text)}</span>'


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


# ---------- tool calls ----------
def args_of(tc):
    a = tc.get("function", {}).get("arguments")
    return maybe_json(a) if isinstance(a, str) else (a or {})


def dx_tables(args):
    out = []
    for key in ("active_diagnoses", "chronic_conditions"):
        rows = [f'<tr><td class="mono">{esc(r.get("icd10"))}</td><td>{esc(r.get("name"))}</td>'
                f'<td>{esc(r.get("acuity"))}</td></tr>' for r in args.get(key) or []]
        rows = rows or ['<tr><td colspan="3" class="muted">none submitted</td></tr>']
        out.append(f'<h3>{esc(key.replace("_", " "))}</h3>' + table(["ICD-10", "name", "acuity"], rows))
    return "".join(out)


def call_html(tc, n):
    name = tc["function"]["name"]
    args = args_of(tc) or {}
    if name == "submit_diagnosis":
        return f'<div class="call submit"><span class="n">#{n}</span><b>{esc(name)}</b>{dx_tables(args)}</div>'
    arg_txt = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return (f'<div class="call"><span class="n">#{n}</span><b>{esc(name)}</b>'
            f'<code>{esc(arg_txt)}</code></div>')


# ---------- tool results ----------
def encounter_html(o):
    head = ", ".join(f"{k}={o[k]}" for k in ("encounter_id", "date", "type", "department", "attending") if k in o)
    rows = [f'<tr><td class="mono">{esc(s.get("section_id"))}</td><td class="mono">{esc(s.get("section_type"))}</td>'
            f'<td class="txt">{esc(s.get("section_text"))}</td></tr>' for s in o["sections"]]
    return f'<p class="muted mono">{esc(head)}</p>' + table(["section id", "type", "text"], rows)


def tool_body(content):
    obj = maybe_json(content)
    if obj == [] or obj == {}:
        return '<p class="muted">Empty result</p>', flag("empty", "warn")
    if isinstance(obj, dict) and isinstance(obj.get("sections"), list):
        return details(f'encounter detail, {len(obj["sections"])} sections', encounter_html(obj)), ""
    text = pretty(obj) if obj is not None else str(content)
    if len(text) <= 500:
        return f"<pre>{esc(text)}</pre>", ""
    preview = re.sub(r"\s+", " ", text)[:90]
    return details(f"{len(text):,} chars: {esc(preview)}", f"<pre>{esc(text)}</pre>"), ""


# ---------- messages ----------
def block(i, role, label, inner, flags=""):
    return (f'<section class="msg {role}"><header><span class="idx">{i}</span>'
            f'<span class="role">{esc(label)}</span>{flags}</header>{inner}</section>')


def render_message(i, m, calls):
    role = m.get("role", "?")
    if role == "tool":
        n, name = calls.get(m.get("tool_call_id"), ("?", "unknown call"))
        body, fl = tool_body(m.get("content"))
        return block(i, "tool", f"result of #{n} {name}", body, fl)
    text = m.get("content") or ""
    flags = ""
    if role == "assistant":
        if re.search(r"</think>\s*$", text):
            text = re.sub(r"\s*</think>\s*$", "", text)
            flags += flag("stray </think>", "warn")
        tcs = m.get("tool_calls") or []
        if len(tcs) > 1:
            flags += flag(f"{len(tcs)} parallel calls")
        body = ""
        if text:
            inner = f'<div class="txt">{esc(text)}</div>'
            body = details(f"reasoning, {len(text):,} chars", inner) if len(text) > 700 else inner
        body += "".join(call_html(tc, calls[tc["id"]][0]) for tc in tcs)
        return block(i, "assistant", "assistant", body, flags)
    inner = f'<div class="txt">{esc(text)}</div>'
    if role == "system":
        inner = details(f"{len(text):,} chars", inner)
    return block(i, role, role, inner)


# ---------- page ----------
TOKENS_LIGHT = ("--bg:#f2f4f7;--surface:#fff;--ink:#18222d;--muted:#5b6978;--line:#d8dee5;--code:#eceff3;"
                "--sys:#7a68b8;--user:#b3741b;--asst:#2c6db0;--tool:#23876a;--warn:#a8650d;--good:#23876a")
TOKENS_DARK = ("--bg:#12161b;--surface:#1a2027;--ink:#e4e9ee;--muted:#93a1b0;--line:#2b343e;--code:#11161c;"
               "--sys:#a597dc;--user:#e0a24a;--asst:#6aa5e0;--tool:#52c4a0;--warn:#e0a24a;--good:#52c4a0")
CSS = """
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);%(L)s}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){%(D)s}}
:root[data-theme="dark"]{%(D)s}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*,*::before,*::after{box-sizing:inherit}
body{margin:0 auto;max-width:62rem;padding:1.2rem 1rem 3rem;background:var(--bg);color:var(--ink);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:1.7rem;margin:0 0 .2rem;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:2rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
h3{font-size:.9rem;margin:.8rem 0 .3rem;color:var(--muted);font-weight:600}
.muted{color:var(--muted)} .mono,code,pre,.idx{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sub{color:var(--muted);margin:0 0 1rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(8rem,1fr));gap:0;border:1px solid var(--line);background:var(--surface)}
.stats div{padding:.6rem .8rem;border-right:1px solid var(--line)} .stats div:last-child{border-right:0}
.stats b{display:block;font-size:1.5rem;line-height:1.2} .stats span{color:var(--muted);font-size:.85rem}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(20rem,1fr));gap:1.2rem}
.scroll{overflow-x:auto} table{border-collapse:collapse;width:100%;background:var(--surface)}
th,td{text-align:left;vertical-align:top;padding:.3rem .6rem;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:.8rem} td.txt{white-space:pre-wrap}
tr.key td{font-weight:700;background:var(--code)}
.bar{display:block;height:.55rem;background:var(--asst);border-radius:2px;min-width:2px}
.meter{height:.5rem;background:var(--code);border-radius:3px;margin-top:.4rem;overflow:hidden}
.meter i{display:block;height:100%;background:var(--asst)}
.msg{--c:var(--muted);border-left:4px solid var(--c);background:var(--surface);margin:.6rem 0;padding:.5rem .9rem .7rem;border-radius:0 8px 8px 0}
.msg.system{--c:var(--sys)} .msg.user{--c:var(--user)} .msg.assistant{--c:var(--asst)} .msg.tool{--c:var(--tool)}
.msg header{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-bottom:.3rem}
.idx{color:var(--muted)} .role{font-weight:600;color:var(--c)}
.flag{font-size:.75rem;padding:0 .45rem;border:1px solid var(--line);border-radius:99px;color:var(--muted)}
.flag.warn{color:var(--warn);border-color:var(--warn)}
.txt{white-space:pre-wrap;word-break:break-word}
.call{margin:.45rem 0 0;padding:.35rem .6rem;background:var(--code);border-radius:4px;display:flex;gap:.6rem;flex-wrap:wrap;align-items:baseline}
.call .n{color:var(--muted)} .call.submit{display:block;border:1px solid var(--good)} .call.submit .n{margin-right:.6rem}
pre{margin:.3rem 0 0;padding:.6rem;background:var(--code);border-radius:4px;white-space:pre-wrap;word-break:break-word;overflow-x:auto}
details.t{margin:.2rem 0} summary{cursor:pointer;color:var(--muted)} summary:focus-visible,button:focus-visible{outline:2px solid var(--asst);outline-offset:2px}
button{font:inherit;padding:.15rem .7rem;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:4px;cursor:pointer}
.tools{display:flex;gap:.5rem;margin:.6rem 0}
"""
JS = ("document.querySelectorAll('[data-all]').forEach(b=>b.onclick=()=>"
      "document.querySelectorAll('details.t').forEach(d=>d.open=b.dataset.all==='1'));")


def render_record(rec, title=None):
    msgs, tools = rec.get("messages", []), rec.get("tools", [])
    meta, metrics = rec.get("meta", {}), rec.get("metrics", {})
    calls, n = {}, 0
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            n += 1
            calls[tc["id"]] = (n, tc["function"]["name"])
    used = Counter(name for _, name in calls.values())
    empties = sum(1 for m in msgs if m.get("role") == "tool" and maybe_json(m.get("content")) in ([], {}))
    budget = meta.get("budget")
    title = title or f'Patient {meta.get("patient_id", "?")}, run {meta.get("episode", "?")}'

    stats = [(rec.get("reward"), "reward"), (len(msgs), "messages"),
             (f"{n}/{budget}" if budget else n, "tool calls"), (empties, "empty results"),
             (meta.get("steps"), "steps"), (meta.get("nudges"), "nudges")]
    stats_html = "".join(f"<div><b>{esc(fmt(v))}</b><span>{esc(l)}</span></div>" for v, l in stats if v is not None)
    meter = f'<div class="meter"><i style="width:{min(100, 100 * n / budget):.0f}%"></i></div>' if budget else ""

    key = meta.get("metric")
    m_rows = [f'<tr class="{"key" if k == key else ""}"><td class="mono">{esc(k)}</td><td>{esc(fmt(v))}</td></tr>'
              for k, v in metrics.items()]
    meta_rows = [f'<tr><td class="mono">{esc(k)}</td><td>{esc(fmt(v))}</td></tr>' for k, v in meta.items()]
    top = max(used.values(), default=1)
    use_rows = [f'<tr><td class="mono">{esc(k)}</td><td>{c}</td><td style="width:40%"><i class="bar" style="width:{100 * c / top:.0f}%"></i></td></tr>'
                for k, c in used.most_common()]

    final = next(((tc, calls[tc["id"]][0]) for m in reversed(msgs) for tc in (m.get("tool_calls") or [])
                  if tc["function"]["name"].startswith("submit")), None)
    answer = call_html(*final) if final else '<p class="muted">No submission found.</p>'

    schema = "".join(details(f'<b>{esc(t["function"]["name"])}</b> '
                             f'<span class="muted">called {used.get(t["function"]["name"], 0)}x. '
                             f'{esc(t["function"].get("description", ""))}</span>',
                             f'<pre>{esc(pretty(t["function"].get("parameters", {})))}</pre>') for t in tools)

    body = f"""<h1>{esc(title)}</h1>
<p class="sub">{esc(meta.get("task", ""))}, model {esc(meta.get("model", "?"))}, split {esc(meta.get("split", "?"))}, ground truth id {esc(meta.get("gt_id", "?"))}</p>
<div class="stats">{stats_html}</div>{meter}
<div class="cols"><div><h2>Metrics</h2>{table(["metric", "value"], m_rows)}</div>
<div><h2>Tool usage</h2>{table(["tool", "calls", ""], use_rows)}<h2>Meta</h2>{table(["field", "value"], meta_rows)}</div></div>
<h2>Submitted answer</h2>{answer}
<h2>Transcript</h2><div class="tools"><button data-all="1">Expand all</button><button data-all="0">Collapse all</button></div>
{"".join(render_message(i, m, calls) for i, m in enumerate(msgs))}
<h2>Tool schema</h2>{schema}"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f'<title>{esc(title)}</title><style>{CSS.replace("%(L)s", TOKENS_LIGHT).replace("%(D)s", TOKENS_DARK)}</style></head>'
            f'<body>{body}<script>{JS}</script></body></html>')


def convert_file(src, num_reports=1):
    """Write HTML for every record in a .json or .jsonl file. Returns the output paths."""
    src = Path(src)
    text = src.read_text(encoding="utf-8")
    out_dir = src.parent.parent / "html_traces"
    
    if src.suffix == ".json":
        recs = json.loads(text)
        recs = recs if isinstance(recs, list) else [recs]
    elif src.suffix == ".jsonl":
        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
        recs = recs[:num_reports]
    else:
        raise ValueError(f"Expected .json or .jsonl input file, but got {src.suffix}")
    
    paths = []
    for k, rec in enumerate(recs):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{src.stem}_{k:03d}.html"
        out_file.write_text(render_record(rec), encoding="utf-8")
        paths.append(out_file)
    
    return paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in convert_file(sys.argv[1], num_reports=int(sys.argv[2])):
        print("wrote", p)
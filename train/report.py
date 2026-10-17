"""Render an episode JSONL (baseline_eval.py --out) as one static HTML page.

    python report.py /runs/baseline.jsonl -o baseline.html --title "Qwen3.5-9B baseline"

Shows the stage funnel, per-turn progress, speedup distribution, timing (generation, env step,
GPU queue wait, eval) and every episode turn by turn: feedback, submitted code, diff vs the
previous turn, raw completion. Output follows the Artifact page contract (no html/head/body).
"""

import argparse
import difflib
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

from baseline_eval import STAGE_ORDER, extract_code, summarize

STAGE_LABEL = {"format": "rejected", "compile": "compile error", "runtime": "runtime error",
               "timeout": "timeout", "correctness": "wrong output", "timed": "correct"}
STAGE_CLASS = {"format": "muted", "compile": "err", "runtime": "err", "timeout": "err",
               "correctness": "warn", "timed": "ok"}
SPEEDUP_BUCKETS = [(0, 0.5, "< 0.5×"), (0.5, 1, "0.5–1×"), (1, 2, "1–2×"), (2, 4, "2–4×"), (4, math.inf, "≥ 4×")]

CSS = """
/* Layout: one reading column; summary → funnel/turns → distribution/timing → episodes */
:root {
  --bg: #f6f7f5; --surface: #ffffff; --fg: #1b211e; --muted: #5d6862; --line: #d9ded9;
  --accent: #0d6b5f; --ok: #1f7a3d; --warn: #a15c00; --err: #b3261e; --mute: #7a847f;
  --add: #e3f3e6; --del: #fbe6e4; --code: #eef1ee;
  --display: "Barlow Semi Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121614; --surface: #1a201d; --fg: #e3e8e4; --muted: #9aa59f; --line: #2c3430;
  --accent: #4fc2ae; --ok: #5ccf7d; --warn: #f0a640; --err: #ff7a6e; --mute: #8b958f;
  --add: #173322; --del: #3a1d1b; --code: #161b18; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #121614; --surface: #1a201d; --fg: #e3e8e4; --muted: #9aa59f; --line: #2c3430;
  --accent: #4fc2ae; --ok: #5ccf7d; --warn: #f0a640; --err: #ff7a6e; --mute: #8b958f;
  --add: #173322; --del: #3a1d1b; --code: #161b18; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 var(--body); }
main { max-width: 1040px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 36px; }
h1, h2 { font-family: var(--display); text-wrap: balance; margin: 0; letter-spacing: .01em; }
h1 { font-size: 2.1rem; font-weight: 600; }
h2 { font-size: 1.3rem; font-weight: 600; }
.sub { color: var(--muted); margin: 4px 0 0; }
.eyebrow { font: 600 .72rem var(--mono); text-transform: uppercase; letter-spacing: .08em; color: var(--accent); }
section { display: grid; gap: 14px; min-width: 0; }
.stats { display: flex; flex-wrap: wrap; gap: 10px 32px; padding: 16px 0; border-block: 1px solid var(--line); }
.stat { display: grid; gap: 2px; }
.stat b { font: 600 1.6rem var(--display); font-variant-numeric: tabular-nums; }
.stat span { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 36px; }
.bars { display: grid; grid-template-columns: minmax(120px, max-content) 1fr 5.5em; gap: 8px 12px; align-items: center; }
.bars .lab { font-size: .9rem; }
.bars .val { font: .85rem var(--mono); text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.track { height: 12px; background: var(--code); border-radius: 2px; overflow: hidden; display: flex; }
.fill { height: 100%; background: var(--accent); }
.fill.ok { background: var(--ok); } .fill.warn { background: var(--warn); }
.fill.err { background: var(--err); } .fill.muted { background: var(--mute); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); vertical-align: middle; }
th { font: 600 .72rem var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
td.num { font-family: var(--mono); font-size: .88rem; }
.scroll { overflow-x: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .82rem; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.chip { display: inline-block; font: 600 .72rem var(--mono); padding: 2px 7px; border-radius: 3px;
  border: 1px solid currentColor; white-space: nowrap; }
.chip.ok { color: var(--ok); } .chip.warn { color: var(--warn); } .chip.err { color: var(--err); } .chip.muted { color: var(--mute); }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
details.ep { background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
details.ep > summary { cursor: pointer; padding: 10px 14px; display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 2fr) 5em; gap: 12px; align-items: center; list-style: none; }
details.ep > summary::-webkit-details-marker { display: none; }
details.ep > summary:focus-visible, details.inner > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.task { font: 500 .9rem var(--mono); overflow-wrap: anywhere; }
.best { font: 600 .95rem var(--mono); text-align: right; }
.best.fast { color: var(--ok); }
.turn { border-top: 1px solid var(--line); padding: 12px 14px; display: grid; gap: 8px; min-width: 0; }
.turnhead { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; font-size: .88rem; color: var(--muted); }
.turnhead strong { color: var(--fg); font-family: var(--display); font-size: 1rem; }
pre { margin: 0; background: var(--code); padding: 10px 12px; border-radius: 3px; overflow-x: auto;
  font: .8rem/1.45 var(--mono); max-height: 420px; }
details.inner > summary { cursor: pointer; font-size: .85rem; color: var(--accent); width: fit-content; }
details.inner[open] > summary { margin-bottom: 6px; }
.d-add { background: var(--add); display: block; } .d-del { background: var(--del); display: block; }
.d-hunk { color: var(--accent); display: block; }
.note { color: var(--muted); font-size: .85rem; }
@media (max-width: 560px) {
  details.ep > summary { grid-template-columns: 1fr 4.5em; }
  details.ep > summary .chips { grid-column: 1 / -1; grid-row: 2; }
}
"""


def pct(a, n):
    return 100 * a / n if n else 0.0


def quantile(vals, q):
    vals = sorted(v for v in vals if v is not None)
    return vals[min(int(q * len(vals)), len(vals) - 1)] if vals else None


def fmt(v, spec=".2f", unit=""):
    return "–" if v is None else f"{v:{spec}}{unit}"


def chip(stage):
    return f'<span class="chip {STAGE_CLASS.get(stage, "muted")}">{escape(STAGE_LABEL.get(stage, stage))}</span>'


def bar_rows(rows):
    """rows: (label, fraction 0..1, value text, css class)."""
    out = ['<div class="bars">']
    for label, frac, val, cls in rows:
        out.append(f'<span class="lab">{escape(label)}</span><span class="track">'
                   f'<span class="fill {cls}" style="width:{100 * frac:.1f}%"></span></span><span class="val">{val}</span>')
    return "".join(out) + "</div>"


def best_speedup(ep):
    return max((t["speedup"] for t in ep["turns"] if t["correct"]), default=None)


def furthest(ep):
    return max((STAGE_ORDER.index(t["stage"]) for t in ep["turns"]), default=0)


def diff_html(prev, cur):
    lines = list(difflib.unified_diff(prev.splitlines(), cur.splitlines(), "previous", "this turn", lineterm="", n=2))
    if not lines:
        return '<p class="note">Identical to the previous submission.</p>'
    cls = lambda ln: "d-hunk" if ln.startswith("@@") else "d-add" if ln.startswith("+") else "d-del" if ln.startswith("-") else ""
    body = "".join(f'<span class="{cls(ln)}">{escape(ln) or " "}</span>' if cls(ln) else escape(ln) + "\n" for ln in lines[2:])
    return f"<pre>{body}</pre>"


def episode_html(ep):
    hist = ep.get("history") or []
    turns_html, prev_code = [], None
    for k, t in enumerate(ep["turns"]):
        h = hist[k] if k < len(hist) else {}
        completion = ep["completions"][k] if k < len(ep.get("completions", [])) else ""
        code = extract_code(completion)
        bits = [f"<strong>Turn {k + 1}</strong>", chip(t["stage"])]
        if t["correct"]:
            bits.append(f'speedup <b>{t["speedup"]:.2f}×</b>')
        if t.get("t_ref_ms") is not None:
            bits.append(f'ref {fmt(t["t_ref_ms"], ".3f", " ms")} → yours {fmt(t.get("t_new_ms"), ".3f", " ms")}')
        if t.get("t_min_ms") is not None:
            bits.append(f'{escape(t.get("bound") or "")}-bound, floor {t["t_min_ms"]:.3f} ms, '
                        f'roofline {fmt(t.get("roofline_frac"), ".0%")} (PyTorch {fmt(t.get("ref_roofline_frac"), ".0%")})')
        if t.get("truncated"):
            bits.append(f'<span class="chip warn">cut off at token limit</span>')
        if t.get("completion_tokens"):
            bits.append(f'{t["completion_tokens"]} tokens')
        if h.get("roofline_violation"):
            bits.append('<span class="chip err">below hardware floor</span>')
        if h.get("eval_s") is not None:
            bits.append(f'eval {h["eval_s"]:.1f} s, GPU queue {h.get("queue_wait_s") or 0:.1f} s')
        if t.get("gen_s") is not None:
            bits.append(f'generation {t["gen_s"]:.0f} s')
        if h.get("suspicious"):
            bits.append('<span class="chip warn">suspicious speedup</span>')
        parts = [f'<div class="turnhead">{" · ".join(bits)}</div>']
        if t.get("feedback"):
            parts.append(f'<pre>{escape(t["feedback"])}</pre>')
        parts.append(f'<details class="inner"><summary>Submitted code ({len(code.splitlines())} lines'
                     f'{", sha " + escape(h["code_sha"]) if h.get("code_sha") else ""})</summary><pre>{escape(code)}</pre></details>')
        if prev_code is not None:
            parts.append(f'<details class="inner"><summary>Diff vs turn {k}</summary>{diff_html(prev_code, code)}</details>')
        parts.append(f'<details class="inner"><summary>Raw completion ({len(completion)} chars)</summary>'
                     f'<pre>{escape(completion)}</pre></details>')
        turns_html.append(f'<div class="turn">{"".join(parts)}</div>')
        prev_code = code
    best = best_speedup(ep)
    eid = f'<span class="note">episode {escape(ep["episode_id"])}</span>' if ep.get("episode_id") else ""
    if ep.get("error"):
        eid += f'<pre>Episode error: {escape(ep["error"])}</pre>'
    return (f'<details class="ep"><summary><span class="task">{escape(ep["task_id"])}</span>'
            f'<span class="chips">{"".join(chip(t["stage"]) for t in ep["turns"])}</span>'
            f'<span class="best{" fast" if best and best > 1 else ""}">{fmt(best, ".2f", "×")}</span></summary>'
            f'<div class="turn">{eid}</div>{"".join(turns_html)}</details>')


def render(eps, title):
    n = len(eps)
    s = summarize(eps)
    fur = [furthest(ep) for ep in eps]
    bests = [best_speedup(ep) for ep in eps]
    idx = STAGE_ORDER.index

    stats = [("episodes", str(n)), ("compiled", f'{s["compile_pct"]:.0f}%'), ("correct", f'{s["correct_pct"]:.0f}%'),
             ("correct on turn 1", f'{s["correct_turn1_pct"]:.0f}%'), ("faster than PyTorch", f'{s["faster_than_ref_pct"]:.0f}%'),
             ("median speedup", fmt(s["median_speedup"], ".2f", "×")),
             ("median roofline", fmt(s.get("median_roofline_frac"), ".0%")), ("suspicious", str(s["suspicious"]))]
    if s.get("samples_per_task"):  # several samples per task: pass@1 / pass@k over tasks
        k = s["samples_per_task"]
        stats[:1] = [("tasks × samples", f'{s["tasks"]} × {k}'), ("pass@1", f'{s["pass@1_pct"]:.0f}%'),
                     (f"pass@{k}", f'{s[f"pass@{k}_pct"]:.0f}%')]

    funnel_counts = [
        ("Submitted", n),
        ("Passed code checks", sum(f >= idx("compile") for f in fur)),
        ("Compiled", sum(f >= idx("runtime") for f in fur)),
        ("Ran to completion", sum(f >= idx("correctness") for f in fur)),
        ("Correct", sum(f == idx("timed") for f in fur)),
        ("Faster than PyTorch", sum(b is not None and b > 1 for b in bests)),
    ]
    funnel = bar_rows([(lab, c / n if n else 0, f"{c} · {pct(c, n):.0f}%", "") for lab, c in funnel_counts])

    max_turns = max((len(ep["turns"]) for ep in eps), default=0)
    turn_rows = []
    for k in range(max_turns):
        at = [ep["turns"][k] for ep in eps if len(ep["turns"]) > k]
        cnt = Counter(t["stage"] for t in at)
        cum = sum(any(t["correct"] for t in ep["turns"][:k + 1]) for ep in eps)
        stack = "".join(f'<span class="fill {STAGE_CLASS[st]}" style="width:{pct(cnt[st], len(at)):.1f}%" '
                        f'title="{STAGE_LABEL[st]}: {cnt[st]}"></span>' for st in STAGE_ORDER if cnt[st])
        gen = next((t.get("gen_s") for t in at if t.get("gen_s") is not None), None)
        turn_rows.append(f'<tr><td class="num">{k + 1}</td><td class="num">{len(at)}</td>'
                         f'<td style="width:45%"><span class="track">{stack}</span></td>'
                         f'<td class="num">{pct(cum, n):.0f}%</td><td class="num">{fmt(gen, ".0f", " s")}</td></tr>')
    legend = "".join(f'<span><i class="fill {STAGE_CLASS[st]}"></i>{STAGE_LABEL[st]}</span>' for st in STAGE_ORDER)

    sp = [b for b in bests if b is not None]
    buckets = [(lab, sum(lo <= b < hi for b in sp)) for lo, hi, lab in SPEEDUP_BUCKETS]
    top = max((c for _, c in buckets), default=0) or 1
    speed = (bar_rows([(lab, c / top, str(c), "ok" if lo >= 1 else "warn") for (lab, c), (lo, _, _) in zip(buckets, SPEEDUP_BUCKETS)])
             if sp else '<p class="note">No correct submissions yet.</p>')

    hist = [h for ep in eps for h in ep.get("history") or [] if h.get("eval_s") is not None]
    gpu_hist = [h for h in hist if h.get("gpu_id") is not None]
    steps = [t.get("step_s") for ep in eps for t in ep["turns"]]
    timing_rows = [
        ("Generation per turn", [t.get("gen_s") for ep in eps for t in ep["turns"]]),
        ("Env step, client side", steps),
        ("GPU evaluation", [h["eval_s"] for h in gpu_hist]),
        ("Waiting for a free GPU", [h.get("queue_wait_s") for h in gpu_hist]),
    ]
    timing = "".join(f'<tr><td>{lab}</td><td class="num">{fmt(quantile(v, .5), ".1f", " s")}</td>'
                     f'<td class="num">{fmt(quantile(v, .95), ".1f", " s")}</td><td class="num">{fmt(max((x for x in v if x is not None), default=None), ".1f", " s")}</td></tr>'
                     for lab, v in timing_rows if any(x is not None for x in v))
    timing = (f'<div class="scroll"><table><tr><th>Phase</th><th>p50</th><th>p95</th><th>max</th></tr>{timing}</table></div>'
              if timing else '<p class="note">No timing fields in this file (written before timing was logged).</p>')

    # roofline: PyTorch vs best correct kernel per task, as share of the hardware floor
    roof_rows = []
    for ep in eps:
        t_best = max((t for t in ep["turns"] if t.get("ref_roofline_frac") is not None),
                     key=lambda t: (t["correct"], t.get("roofline_frac") or 0), default=None)
        if t_best:
            mine = t_best.get("roofline_frac") if t_best["correct"] else None
            roof_rows.append(f'<tr><td class="task">{escape(ep["task_id"])}</td><td>{escape(t_best.get("bound") or "–")}</td>'
                             f'<td class="num">{fmt(t_best.get("t_min_ms"), ".3f", " ms")}</td>'
                             f'<td class="num">{fmt(t_best["ref_roofline_frac"], ".0%")}</td>'
                             f'<td class="num">{fmt(mine, ".0%")}</td>'
                             f'<td class="num">{fmt(1 / t_best["ref_roofline_frac"] if t_best["ref_roofline_frac"] else None, ".1f", "×")}</td></tr>')
    roof_html = (f'<div class="scroll"><table><tr><th>Task</th><th>Bound</th><th>Floor</th><th>PyTorch</th><th>Best kernel</th>'
                 f'<th>Max speedup</th></tr>{"".join(roof_rows)}</table></div><p class="note">Share of the roofline floor '
                 f'(max of FLOPs ÷ peak compute, bytes ÷ peak bandwidth). Max speedup = how much faster than PyTorch any kernel could be.</p>'
                 if roof_rows else '<p class="note">No roofline data (file predates it, or the GPU is not in the peak table).</p>')

    order = sorted(eps, key=lambda ep: (-furthest(ep), -(best_speedup(ep) or 0), ep["task_id"]))
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{lab}</span></div>' for lab, v in stats)
    return f"""<title>{escape(title)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=JetBrains+Mono:wght@400;500;600&family=Source+Sans+3:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
<header><div class="eyebrow">kernel_env run report</div><h1>{escape(title)}</h1>
<p class="sub">{n} KernelBench episodes, up to {max_turns} submissions each. Speedup is eager PyTorch time divided by kernel time on the eval GPU.</p></header>
<div class="stats">{stat_html}</div>
<div class="two">
<section><h2>Where episodes stopped</h2>{funnel}<p class="note">Best stage each episode reached in any turn.</p></section>
<section><h2>Best speedup, correct episodes</h2>{speed}</section>
</div>
<section><h2>Turn by turn</h2><div class="scroll"><table><tr><th>Turn</th><th>Submissions</th><th>Outcome mix</th><th>Correct so far</th><th>Generation</th></tr>{"".join(turn_rows)}</table></div>
<div class="legend">{legend}</div></section>
<section><h2>Distance to the hardware roof</h2>{roof_html}</section>
<section><h2>Where the time went</h2>{timing}</section>
<section><h2>Episodes</h2><p class="note">Sorted by furthest stage, then speedup. Open one to see each submission, its feedback and the diff from the previous turn.</p>
{"".join(episode_html(ep) for ep in order)}</section>
</main>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episodes", help="JSONL from baseline_eval.py --out")
    p.add_argument("-o", "--out", default="report.html")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    eps = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
    Path(args.out).write_text(render(eps, args.title or Path(args.episodes).stem))
    print(f"{len(eps)} episodes -> {args.out}")


if __name__ == "__main__":
    main()

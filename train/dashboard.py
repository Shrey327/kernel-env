"""Live, loopback-only viewer for a small_run.py experiment; no extra GPU required.

python dashboard.py --run-dir ../runs/proof-20261019/results --remote-dir /runs/proof-20261019-7b \
    --container ta-... --log /tmp/kernel-proof-train.log --billing /tmp/kernel-proof-billing-progress.json
"""

import argparse
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from baseline_eval import episode_return, summarize
from report import CSS, episode_html

# Read artifacts only: never import or execute generated kernels in the viewer.
READ = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
def read(name, lines=False):
    f=p/name
    if not f.exists(): return [] if lines else {}
    if not lines: return json.loads(f.read_text())
    out=[]
    for line in f.read_text().splitlines():
        try: out.append(json.loads(line))
        except ValueError: pass
    return out
states=sorted((p/'adapter').glob('checkpoint-*/trainer_state.json'),key=lambda x:int(x.parent.name.split('-')[-1]))
print(json.dumps({'manifest':read('manifest.json'),'provenance':read('provenance.json'),
 'before':read('before.jsonl',True),'train':read('adapter/rollouts-rank0.jsonl',True),
 'after':read('after.jsonl',True),'comparison':read('comparison.json'),
 'trainer':json.loads(states[-1].read_text()) if states else {}}))
'''


def view(snapshot, log_text="", cost=None, episode_limit=0):
    manifest = snapshot.get("manifest", {})
    events = []
    for line in log_text.splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict) and event.get("event") == "step":
                events.append(event)
        except ValueError:
            pass
    goal = manifest.get("steps", 16)
    progress = max([int(n) for n in re.findall(r"(\d+)/" + str(goal) + r"\s*\[", log_text)]
                   + [snapshot.get("trainer", {}).get("global_step", 0)])
    train = snapshot.get("train", [])
    by_step = defaultdict(list)
    for ep in train:
        by_step[ep["step"] + 1].append(ep)
    metrics = {int(m["step"]): m for m in snapshot.get("trainer", {}).get("log_history", []) if "step" in m}
    steps = []
    for step, eps in sorted(by_step.items()):
        s = summarize(eps)
        steps.append({"step": step, "episodes": len(eps), "correct": s["correct_pct"],
                      "task": escape(", ".join(sorted({e["task_id"].split("/")[-1] for e in eps}))),
                      "first": s["correct_turn1_pct"], "mean_return": sum(
                          e.get('learning_return', episode_return([t["reward"] for t in e["turns"]], "discounted", 0.4))
                          for e in eps) / len(eps), "updated": step <= progress,
                      "grad_norm": metrics.get(step, {}).get("grad_norm"),
                      **{k: metrics.get(step, {}).get(k) for k in ("loss", "epoch", "learning_rate", "entropy")}})
    before, after = snapshot.get("before", []), snapshot.get("after", [])
    phase = "Complete" if snapshot.get("comparison") else "Final evaluation" if after else (
        "Training" if train else "Baseline")
    if not snapshot.get("comparison") and "PHASE after:" in log_text:
        phase = "Final evaluation"
    groups = {}
    for name, ids in manifest.get("groups", {}).items():
        groups[name] = {label: summarize([e for e in eps if e["task_id"] in ids])
                        for label, eps in [("before", before), ("after", after)]}
    sections = {}
    for key, eps in [("before", before), ("train", train), ("after", after)]:
        parts = []
        start = max(0,len(eps)-episode_limit) if episode_limit else 0
        for i, ep in enumerate(eps[start:],start):
            label = f'Step {ep["step"] + 1} · episode {i + 1}' if key == "train" else f'Episode {i + 1}'
            body = episode_html(ep).replace('<details class="ep">', f'<details class="ep" id="{key}-{i}">', 1)
            for j in range(body.count('<details class="inner">')):
                body = body.replace('<details class="inner">', f'<details class="inner" id="{key}-{i}-detail-{j}">', 1)
            parts.append(f'<article><p class="note">{escape(label)}</p>{body}</article>')
        sections[key] = "".join(parts) or '<p class="note">No episodes yet. This updates automatically.</p>'
    return {"phase": phase, "updated": time.time(), "steps": steps, "completed": progress, "goal": goal,
            "kernel_evals": len(events) if events else sum(len(e["turns"]) for k in ("before", "train", "after") for e in snapshot.get(k, [])),
            "activity": (f'{events[-1]["task_id"]} · attempt {events[-1]["turn"]} · {events[-1]["stage"]}' if events else "Collecting the first submissions…"),
            "cost": cost, "before": summarize(before), "after": summarize(after), "sections": sections,
            "comparison": snapshot.get("comparison"), "model": manifest.get("model", "Loading…"),
            "groups": groups, "metrics": metrics, "warning": None,
            "episodes_per_update": manifest.get("episodes_per_update",4),
            "budget_cap_usd":manifest.get("budget_cap_usd",50)}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kernel Lab · Live experiment</title><style>__CSS__

/* Shared dashboard layout and controls. */
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1200px;gap:28px;padding-top:40px;padding-bottom:64px}main>*{min-width:0}h1{font-size:clamp(28px,3.8vw,42px);letter-spacing:-.035em;line-height:1.15}h2{font-size:21px;letter-spacing:-.02em}h3{line-height:1.4}p{margin:0}.sub{line-height:1.65;margin-top:8px;font-size:14px}.eyebrow{margin-bottom:12px;letter-spacing:.12em}
.top{display:flex;justify-content:space-between;align-items:start;gap:20px}header.top{padding-bottom:4px}header.top>div:last-child{flex-shrink:0;text-align:right}.live{display:inline-flex;align-items:center;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:var(--accent);font-size:12px;font-weight:600}#stamp{margin-top:8px}
a{color:var(--accent);text-underline-offset:3px;text-decoration-thickness:1px}a:hover{text-decoration-thickness:2px}a:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}button,select,input{font:inherit;border:1px solid var(--line);background:var(--surface);color:var(--fg);padding:9px 12px;border-radius:7px;min-height:40px}button{cursor:pointer;font-size:13px;font-weight:600}button:hover{border-color:var(--accent)}button[aria-selected="true"]{color:var(--accent);border-color:var(--accent);background:var(--code)}input::placeholder{color:var(--muted)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.toolbar input{margin-left:auto;max-width:260px}.toolbar[aria-label="Pages"]{gap:8px 18px;padding:16px 0;border-block:1px solid var(--line);font-size:13px}.toolbar[aria-label="Pages"] a{padding:5px 0;text-decoration:none;font-weight:550}.toolbar[aria-label="Pages"] a:hover{text-decoration:underline}.toolbar label{display:flex;align-items:center;gap:8px;margin-left:auto;color:var(--muted);white-space:nowrap}.toolbar label select{min-height:34px;padding:5px 10px}.toolbar[role="tablist"]{padding:8px;background:var(--code);border-radius:10px}.toolbar[role="tablist"] button{border-color:transparent}.toolbar[role="tablist"] button[aria-selected="true"]{border-color:var(--line);background:var(--surface)}
.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:0;border:1px solid var(--line);background:var(--surface);padding:0;border-radius:12px;overflow:hidden}.stat{min-width:0;padding:20px;border-right:1px solid var(--line);gap:8px}.stat:last-child{border-right:0}.stat b{font:650 clamp(22px,2.4vw,30px)/1.1 system-ui,sans-serif;letter-spacing:-.04em;overflow-wrap:anywhere}.stat span{font-size:10px;line-height:1.6;letter-spacing:.07em}
.pipeline{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.phase{padding:12px 14px;border:1px solid var(--line);border-top:3px solid var(--line);border-radius:7px;font-size:12px;line-height:1.5;color:var(--muted)}.phase.active{border-top-color:var(--accent);background:var(--surface);color:var(--accent);font-weight:650}.phase.done{border-top-color:var(--ok);color:var(--ok)}
.panel{background:var(--surface);border:1px solid var(--line);padding:24px;border-radius:12px;min-width:0}.two{gap:20px;grid-template-columns:repeat(2,minmax(0,1fr))}.panel>h2{margin-bottom:2px}.traces{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:16px}.traces .panel{padding:20px}.traces h3{font-size:13px;font-weight:600;margin:0}.trace-label{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.trace-label select{max-width:180px;font-size:11px;padding:5px 8px;min-height:32px}svg{display:block;width:100%;height:155px;color:var(--muted);overflow:visible}.traces svg{height:190px;margin-top:12px}.traces svg text{font-family:ui-monospace,monospace}svg circle:focus-visible{outline:2px solid var(--fg)}
.stepgrid{display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:5px}.cell{background:var(--code);padding:8px 0;text-align:center;font:11px ui-monospace,monospace;border-radius:4px;border-bottom:3px solid var(--line);color:var(--muted)}.cell.solved{background:var(--add);border-color:var(--ok);color:var(--ok)}.cell.partial{border-color:var(--warn)}.progress{height:5px;background:var(--code);margin-top:8px;border-radius:3px;overflow:hidden}.progress>div{height:100%;background:var(--accent)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface);overscroll-behavior-x:contain}th,td{padding:13px 14px;font-size:13px;line-height:1.5}th{font-size:10px;background:var(--code);white-space:nowrap;letter-spacing:.06em}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:var(--code)}td.num{font-size:12px}.compare td{padding:13px 10px}.compare th{padding:10px}.compare td:first-child{font-weight:550}.scroll table{min-width:680px}
#episodes{display:grid;gap:12px}article .note{margin:0 0 7px}details.ep{border-radius:10px;overflow:hidden}details.ep>summary{padding:15px 18px;gap:14px}details.ep>summary:hover{background:var(--code)}details.ep>summary:before{content:"+";color:var(--accent);font:18px ui-monospace,monospace;position:absolute;left:16px}details.ep[open]>summary:before{content:"−"}details.ep>summary{position:relative;padding-left:40px}details.ep[open]{border-color:var(--accent)}.task{font-size:12px;line-height:1.6}.chip{font-size:10px;border-radius:4px;padding:3px 6px}.chips{gap:5px}.turn{padding:18px;gap:12px}.turnhead{font-size:12px;line-height:1.7}.turnhead strong{font:650 13px system-ui,sans-serif}details.inner>summary{font-size:12px;padding:4px 0}pre{padding:15px;border:1px solid var(--line);border-radius:7px;font-size:11px;line-height:1.75;max-height:440px;tab-size:4}
.small,.note{font-size:12px;line-height:1.75;color:var(--muted)}#activity,#warning{margin-top:-14px}#warning{color:var(--warn);overflow-wrap:anywhere}#warning:empty,#activity:empty{display:none}footer{padding-top:22px;border-top:1px solid var(--line);line-height:1.8}
@media(max-width:800px){.stats{grid-template-columns:repeat(3,minmax(0,1fr))}.stat{border-bottom:1px solid var(--line);padding:18px}.stat:nth-child(3){border-right:0}.stat:nth-child(n+4){border-bottom:0}.two{grid-template-columns:1fr}.toolbar label{margin-left:0}}
@media(max-width:560px){main{padding-top:24px;gap:24px}header.top{flex-direction:column}header.top>div:last-child{text-align:left}.top{flex-wrap:wrap;gap:12px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.stat:nth-child(3){border-right:1px solid var(--line)}.stat:nth-child(even){border-right:0}.stat:nth-child(4){border-bottom:1px solid var(--line)}.stat:last-child{grid-column:1/-1}.stat b{font-size:25px}.panel{padding:18px}.pipeline{gap:4px}.phase{padding:9px 5px;font-size:10px;overflow-wrap:anywhere}.stepgrid{grid-template-columns:repeat(8,minmax(0,1fr))}.toolbar input{margin-left:0;width:100%;max-width:none}.toolbar[aria-label="Pages"]{gap:8px 16px}.toolbar label{width:100%;padding-top:6px}.trace-label{align-items:start}.trace-label select{max-width:100%}details.ep>summary{grid-template-columns:minmax(0,1fr) 4em;padding:14px 14px 14px 36px}details.ep>summary:before{left:12px}details.ep>summary .chips{grid-column:1/-1;grid-row:2}.turn{padding:14px}.bars{grid-template-columns:minmax(75px,max-content) 1fr 4em;gap:8px}.bars .lab,.bars .val{font-size:11px}.traces .panel{padding:16px}}

/* Monochrome surfaces and controls. */
:root{--bg:#f6f6f6;--surface:#fff;--fg:#171717;--ink:#171717;--muted:#626262;--line:#dedede;--accent:#262626;--ok:#333;--warn:#666;--err:#222;--mute:#777;--code:#f0f0f0;--add:#e9e9e9;--del:#e1e1e1;--green:#ebebeb;--purple:#555;--selected:#202020;--on-selected:#fff;--chart:#262626;--chart-alt:#777;--chart-low:#a1a1a1;color-scheme:light}
:root[data-theme="light"]{--bg:#f6f6f6;--surface:#fff;--fg:#171717;--ink:#171717;--muted:#626262;--line:#dedede;--accent:#262626;--ok:#333;--warn:#666;--err:#222;--mute:#777;--code:#f0f0f0;--add:#e9e9e9;--del:#e1e1e1;--green:#ebebeb;--purple:#555;--selected:#202020;--on-selected:#fff;--chart:#262626;--chart-alt:#777;--chart-low:#a1a1a1;color-scheme:light}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#101010;--surface:#191919;--fg:#f2f2f2;--ink:#f2f2f2;--muted:#a3a3a3;--line:#333;--accent:#ededed;--ok:#dedede;--warn:#b1b1b1;--err:#f2f2f2;--mute:#999;--code:#222;--add:#303030;--del:#383838;--green:#292929;--purple:#c4c4c4;--selected:#ededed;--on-selected:#171717;--chart:#ededed;--chart-alt:#a3a3a3;--chart-low:#696969;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101010;--surface:#191919;--fg:#f2f2f2;--ink:#f2f2f2;--muted:#a3a3a3;--line:#333;--accent:#ededed;--ok:#dedede;--warn:#b1b1b1;--err:#f2f2f2;--mute:#999;--code:#222;--add:#303030;--del:#383838;--green:#292929;--purple:#c4c4c4;--selected:#ededed;--on-selected:#171717;--chart:#ededed;--chart-alt:#a3a3a3;--chart-low:#696969;color-scheme:dark}
body{background:var(--bg);color:var(--fg);letter-spacing:-.01em}h1{font-weight:700;letter-spacing:-.045em}h2{font-weight:650}::selection{background:var(--selected);color:var(--on-selected)}a{color:var(--accent)}.eyebrow{color:var(--muted)}.panel,.card{box-shadow:0 1px 2px #00000006;border-radius:10px}.callout{border-left-color:var(--accent)}.audit{border-left-color:var(--muted)}.diagram .arrow{stroke:var(--accent)}.diagram .gpu{stroke:var(--muted)}.number{color:var(--fg)}.tag,.status{color:var(--fg);border-color:var(--line);background:var(--code)}nav[aria-label="Pages"]{gap:8px}nav[aria-label="Pages"] a,nav[aria-label="Pages"] strong{padding:8px 12px;border-radius:6px}nav[aria-label="Pages"] a:hover{background:var(--code);text-decoration:none}nav[aria-label="Pages"] [aria-current="page"]{background:var(--selected);color:var(--on-selected)}.jump a{color:var(--muted)}.jump a:hover{color:var(--fg)}.flow div{border-top-color:var(--fg)}.flow b{color:var(--muted)}.timeline:before,.milestone:before{background:var(--fg)}footer a{color:var(--fg)}

.live{background:var(--code);color:var(--fg);border-color:var(--line);font-weight:650}.stats{border-color:var(--line);box-shadow:0 1px 2px #00000006}.stat b{font-weight:650;color:var(--fg)}.stat:first-child{background:var(--code)}.stat span{color:var(--muted)}.phase{background:transparent;color:var(--muted);border-top-color:var(--line)}.phase.active{background:var(--code);color:var(--fg);border-top-color:var(--fg)}.phase.done{color:var(--fg);border-top-color:var(--muted)}.toolbar[aria-label="Pages"] a:first-child{background:var(--selected);color:var(--on-selected)}.toolbar[aria-label="Pages"] a:first-child:hover{background:var(--selected)}.toolbar[aria-label="Pages"] a{border:1px solid transparent}.toolbar[role="tablist"]{background:var(--code);border:1px solid var(--line)}button[aria-selected="true"],.toolbar[role="tablist"] button[aria-selected="true"]{background:var(--selected);border-color:var(--selected);color:var(--on-selected)}button:hover,select:hover{border-color:var(--muted)}.cell.solved{background:var(--selected);color:var(--on-selected);border-color:var(--selected)}.cell.partial{background:var(--code);border-bottom:3px dashed var(--muted)}.chip.ok{color:var(--fg);border-style:solid}.chip.warn{color:var(--muted);border-style:dashed}.chip.err{color:var(--fg);border-style:double}.chip.muted{color:var(--muted)}.fill,.fill.ok,.fill.warn,.fill.err,.fill.muted{background:var(--chart)}.fill.warn{opacity:.65}.fill.err{opacity:.4}.fill.muted{opacity:.25}details.ep[open]{border-color:var(--muted)}.best.fast,#warning{color:var(--fg)}.progress>div{background:var(--fg)}
#rewardChart polyline,#rewardChart circle,#trainerChart polyline,#chart rect{stroke:var(--chart)}#rewardChart circle,#chart rect{fill:var(--chart)}#rewardChart polyline,#trainerChart polyline{stroke-width:2.5}#lossChart polyline,#lossChart circle,#evaluatorChart polyline{stroke:var(--chart-alt)}#lossChart circle{fill:var(--chart-alt)}#lossChart polyline,#evaluatorChart polyline{stroke-dasharray:6 4}#metricChart polyline,#metricChart circle{stroke:var(--chart)}#metricChart polyline{stroke-dasharray:2 4}#metricChart circle{fill:var(--chart)}#trainerChart polyline:nth-of-type(2),#evaluatorChart polyline:nth-of-type(2){stroke:var(--chart-low);stroke-dasharray:3 5}svg line{stroke:var(--line)}.legend i{background:var(--muted)!important}
</style><main><header class="top"><div><div class="eyebrow">Kernel Lab / OpenEnv × GRPO</div><h1>A small run. Every step visible.</h1><p class="sub" id="model">Connecting to experiment…</p></div><div><div class="live" id="live" role="status">● Connecting</div><div class="small" id="stamp"></div></div></header>
<div class="pipeline" id="pipeline"></div>
<div class="stats"><div class="stat"><b id="completed">–</b><span>Training updates</span></div><div class="stat"><b id="evals">–</b><span>Kernel submissions</span></div><div class="stat"><b id="base">–</b><span>Baseline correct</span></div><div class="stat"><b id="after">Pending</b><span>After RL, correct</span></div><div class="stat"><b id="cost">–</b><span>Reported spend / $50 cap</span></div></div><p class="small" id="activity" role="status"></p>
<p class="small" id="warning" role="status"></p>
<section><div class="top"><div><h2>Optimizer traces</h2><p class="sub">Recorded metrics for each update. Hover over a point for its exact value and epoch.</p></div><a href="/metrics.json" download="training-metrics.json">Download all metrics ↓</a></div><div class="traces"><div class="panel"><h3>Reward / discounted episode return</h3><svg id="rewardChart" viewBox="0 0 350 190" role="img" aria-label="Reward by optimizer step"></svg></div><div class="panel"><h3>GRPO policy loss</h3><svg id="lossChart" viewBox="0 0 350 190" role="img" aria-label="GRPO loss by optimizer step"></svg></div><div class="panel"><div class="trace-label"><h3>More traces</h3><select id="metric" aria-label="Additional training metric"><option value="learning_rate">Learning rate</option><option value="grad_norm">Gradient norm</option><option value="entropy">Policy entropy</option><option value="reward_std">Reward standard deviation</option><option value="kernel/tied_episodes">Episodes tied for timing noise</option><option value="kernel/raw_return_std">Raw return spread</option><option value="kernel/learning_return_std">Learning return spread</option><option value="epoch">Epoch</option><option value="completions/mean_length">Completion tokens</option><option value="clip_ratio/region_mean">Clipped-token ratio</option><option value="step_time">Seconds per step</option><option value="num_tokens">Tokens processed</option></select></div><svg id="metricChart" viewBox="0 0 350 190" role="img" aria-label="Selected training metric by optimizer step"></svg></div></div><p class="small">X axis: optimizer step. Groups use different training tasks. A lower policy loss or higher training reward alone does not establish better held-out kernels.</p></section>
<div class="two"><section class="panel"><h2>How training is progressing</h2><p class="note">Each step samples four episodes of two attempts. Inverted cells = all episodes solved; dashed borders = some or none. Hover over a cell for its update status.</p><div class="stepgrid" id="stepgrid"></div><svg id="chart" viewBox="0 0 640 155" role="img" aria-label="Correct training episodes by optimizer step"></svg><div class="small">Bar height: correct episodes in that training group. This is training performance, not held-out progress.</div></section>
<section class="panel"><h2>The comparison that matters</h2><p class="note">Same tasks, model family, GPU and greedy decoding. Each task gets two attempts.</p><table class="compare"><thead><tr><th>Task group</th><th>Base</th><th>After RL</th></tr></thead><tbody id="comparison"></tbody></table><p class="small">Four training tasks, three reserved transfer tasks, three official held-out tasks. A ten-task pilot cannot establish broad generalization.</p></section></div>
<section><h2>Training steps</h2><div class="scroll"><table><thead><tr><th>Step</th><th>Epoch</th><th>Task</th><th>State</th><th>Correct episodes</th><th>Turn 1 correct</th><th>Mean return</th><th>Loss</th><th>Learning rate</th><th>Gradient norm</th></tr></thead><tbody id="steps"></tbody></table></div><p class="small">Groups can contain different tasks, so bar heights are not a learning curve on a fixed task. Trainer traces sync from saved checkpoint history. Final evaluation measures improvement.</p></section>
<section><div class="top"><div><h2>Inspect every attempt</h2><p class="sub">Open an episode for code, feedback, runtime and a diff between turns.</p></div><button id="refresh">Refresh now</button></div><div class="toolbar" role="tablist" aria-label="Experiment phase"><button data-phase="before" role="tab">Baseline</button><button data-phase="train" role="tab" aria-selected="true">Training</button><button data-phase="after" role="tab">After RL</button><input id="search" aria-label="Filter episodes by task name" placeholder="Filter by task name…"></div><div id="episodes"></div></section>
<footer class="small">Private live view on this computer · refreshes every 10 seconds · cost reporting can lag · <a href="https://app.notion.com/p/3fcecae370058136a9a5ef9dcc0457a1" target="_blank" rel="noopener">Build notes and findings ↗</a></footer></main>
<script>
let current=null,phase='train',lastHtml='';const $=id=>document.getElementById(id);const n=x=>x==null?'–':x.toFixed(2);const score=s=>s&&s.episodes?Math.round(s.correct_pct*s.episodes/100)+' / '+s.episodes:'Pending';
const numeric=x=>x==null?'–':Math.abs(x)>0&&Math.abs(x)<0.001?x.toExponential(2):x.toFixed(4);
function trace(id,key,color){let rows=Object.values(current.metrics).filter(m=>typeof m[key]==='number'&&Number.isFinite(m[key])).sort((a,b)=>a.step-b.step);if(!rows.length){$(id).innerHTML='<text x="20" y="90" fill="currentColor" font-size="12">Waiting for trainer metrics…</text>';return}let lo=Math.min(...rows.map(r=>r[key])),hi=Math.max(...rows.map(r=>r[key]));let span=hi-lo||Math.max(Math.abs(hi)*0.1,0.001);lo-=span*.12;hi+=span*.12;let x=s=>48+(s-1)*285/Math.max(current.goal-1,1),y=v=>145-(v-lo)*125/(hi-lo);let axes='';for(let i=0;i<3;i++){let v=lo+(hi-lo)*i/2,yy=y(v);axes+='<line x1="48" x2="335" y1="'+yy+'" y2="'+yy+'" stroke="#7c8b82" stroke-opacity=".25"/><text x="2" y="'+(yy+4)+'" fill="currentColor" font-size="10">'+numeric(v)+'</text>'}let points=rows.map(r=>x(r.step)+','+y(r[key])).join(' ');$(id).innerHTML=axes+'<polyline points="'+points+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>'+rows.map(r=>'<circle tabindex="0" cx="'+x(r.step)+'" cy="'+y(r[key])+'" r="4" fill="'+color+'"><title>Step '+r.step+' · epoch '+n(r.epoch)+' · '+key+': '+r[key]+'</title></circle>').join('')+'<text x="48" y="169" fill="currentColor" font-size="11">1</text><text x="320" y="169" fill="currentColor" font-size="11">'+current.goal+'</text><text x="145" y="184" fill="currentColor" font-size="10">Optimizer step</text>'}
function traces(){if(!current)return;trace('rewardChart','reward','#4fc2ae');trace('lossChart','loss','#b29bf7');trace('metricChart',$('metric').value,'#e8ae58')}
function episodes(){if(!current)return;let h=current.sections[phase];if(h!==lastHtml){let opened=[...document.querySelectorAll('details[open][id]')].map(e=>e.id);$('episodes').innerHTML=h;opened.forEach(id=>{let e=$(id);if(e)e.open=true});lastHtml=h}let q=$('search').value.toLowerCase();document.querySelectorAll('#episodes article').forEach(e=>e.hidden=!e.querySelector('.task').textContent.toLowerCase().includes(q));document.querySelectorAll('[data-phase]').forEach(e=>e.setAttribute('aria-selected',e.dataset.phase===phase))}
function render(d){current=d;$('model').textContent=d.model+' · LoRA rank 16 · 2× H100 · 2-turn episodes';$('live').textContent='● '+d.phase;$('stamp').textContent='Synced '+new Date(d.updated*1000).toLocaleTimeString();$('completed').textContent=d.completed+' / '+d.goal;$('evals').textContent=d.kernel_evals;$('base').textContent=score(d.before);$('after').textContent=score(d.after);$('cost').textContent=d.cost==null?'Pending':'$'+d.cost.toFixed(2)+' / $50';$('warning').textContent=d.warning||'';$('activity').textContent='Latest evaluator feedback: '+d.activity;
let p=['GPU checks','Baseline','GRPO training','Final evaluation','Results'];let idx=d.phase==='Complete'?4:d.phase==='Final evaluation'?3:d.phase==='Training'?2:1;$('pipeline').innerHTML=p.map((s,i)=>'<div class="phase '+(i<idx?'done':i===idx?'active':'')+'">'+(i<idx?'✓ ':i===idx?'● ':'')+s+'</div>').join('');
let by=new Map(d.steps.map(s=>[s.step,s]));$('stepgrid').innerHTML=Array.from({length:d.goal},(_,i)=>{let s=by.get(i+1);return '<div class="cell '+(s&&s.updated?(s.correct===100?'solved':'partial'):'')+'" title="Step '+(i+1)+(s?': '+s.correct+'% correct'+(s.updated?', update complete':', collecting rollout'):', pending')+'">'+(i+1)+'</div>'}).join('');
let chart='<line x1="25" y1="125" x2="635" y2="125" stroke="#aaa"/><text x="0" y="20" fill="currentColor" font-size="11">100%</text><text x="0" y="126" fill="currentColor" font-size="11">0</text>';$('chart').innerHTML=chart+d.steps.map(s=>{let x=35+(s.step-1)*36,w=24,h=s.correct;return '<rect x="'+x+'" y="'+(125-h)+'" width="'+w+'" height="'+h+'" fill="'+(s.updated?'#0d6b5f':'#aab8b1')+'"/><text x="'+(x+8)+'" y="145" fill="currentColor" font-size="10">'+s.step+'</text>'}).join('');
$('steps').innerHTML=d.steps.map(s=>'<tr><td>'+s.step+'</td><td>'+n(s.epoch)+'</td><td>'+s.task+'</td><td>'+(s.updated?'✓ Updated':'Rollout collected')+'</td><td>'+Math.round(s.correct*s.episodes/100)+' / '+s.episodes+'</td><td>'+s.first.toFixed(0)+'%</td><td>'+n(s.mean_return)+'</td><td>'+numeric(s.loss)+'</td><td>'+numeric(s.learning_rate)+'</td><td>'+numeric(s.grad_norm)+'</td></tr>').join('')||'<tr><td colspan="10">Waiting for the first rollout group.</td></tr>';
let groups=[['training_tasks','Training tasks'],['reserved_transfer','Reserved transfer'],['official_heldout','Official held-out']];$('comparison').innerHTML=groups.map(([k,label])=>{let c=d.groups[k];return '<tr><td>'+label+'</td><td>'+(c?score(c.before):'Pending')+'</td><td>'+(c?score(c.after):'Pending')+'</td></tr>'}).join('');episodes();traces()}
async function refresh(){try{let r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error('Connection failed');render(await r.json())}catch(e){$('warning').textContent='Live connection unavailable. Last successful snapshot remains visible.'}}
document.querySelectorAll('[data-phase]').forEach(e=>e.onclick=()=>{phase=e.dataset.phase;lastHtml='';episodes()});$('search').oninput=episodes;$('metric').onchange=traces;$('refresh').onclick=refresh;refresh();setInterval(refresh,10000);
</script></html>'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--remote-dir", required=True)
    p.add_argument("--container", default="")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--billing", type=Path, required=True)
    p.add_argument("--app-id", action="append", default=[])
    p.add_argument("--port", type=int, default=8099)
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache = args.run_dir / "live-snapshot.json"
    snapshot = json.loads(cache.read_text()) if cache.exists() else {}
    state = view(snapshot)
    synced_at = cache.stat().st_mtime if cache.exists() else time.time()

    def collect():
        nonlocal snapshot, state, synced_at
        while True:
            warning = None
            try:
                if (args.run_dir / "comparison.json").exists():
                    r = subprocess.run(["python3", "-c", READ, str(args.run_dir)], capture_output=True, text=True, check=True)
                else:
                    r = subprocess.run(["modal", "container", "exec", args.container, "--", "python", "-c", READ,
                                        args.remote_dir], capture_output=True, text=True, check=True, timeout=25)
                snapshot = json.loads(r.stdout)
                cache.write_text(json.dumps(snapshot))
                synced_at = time.time()
            except (OSError, ValueError, subprocess.SubprocessError):
                warning = "Cloud sync unavailable; showing the last snapshot. Final artifacts will replace it when downloaded."
            cost = None
            if args.billing.exists():
                try:
                    rows = json.loads(args.billing.read_text())
                    cost = sum(float(r["Cost"]) for r in rows if r["Object ID"] in args.app_id)
                except ValueError:
                    cost = state.get("cost")
            state = view(snapshot, args.log.read_text() if args.log.exists() else "", cost)
            state["updated"] = synced_at
            state["warning"] = warning
            time.sleep(20)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/state", "/metrics.json"):
                self.send_error(404)
                return
            body = (json.dumps(state["metrics"] if self.path == "/metrics.json" else state).encode()
                    if self.path != "/" else PAGE.replace("__CSS__", CSS).encode())
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path != "/" else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    threading.Thread(target=collect, daemon=True).start()
    print(f"Kernel Lab: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

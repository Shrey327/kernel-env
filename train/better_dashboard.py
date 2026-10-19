"""New-run reader reuses the existing reader's token check; no new credential is created."""
import hashlib
import asyncio
import math
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse,Response
from campaign_dashboard import state_from,read_json,read_rows
from baseline_eval import summarize

EXISTING_READER='https://shreyansh-pathak27--kernel-env-live-dashboard-live-dashboard.us-east.modal.run'
SERVICE_NAME='kernel-env-better-dashboard'
READER_APP_ID='ap-sXAmPnyHHyFO65EBoXbkIm'


def lease_expired(policy,now,terminal=False):
    created,expires=policy.get('created_unix'),policy.get('expires_unix')
    return (terminal or policy.get('service_name')!=SERVICE_NAME or
            not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (created,expires)) or
            not 0<expires-created<=6*3600 or now>=expires or policy.get('max_cost_usd')!=0.5)


async def stop_owned_reader(policy):
    import modal
    from modal.client import _Client
    from modal_proto import api_pb2
    if policy.get('app_id')!=READER_APP_ID:
        raise RuntimeError('reader policy identity changed; refuse to stop another app')
    app=await modal.App.lookup.aio(SERVICE_NAME,create_if_missing=False)
    if app.app_id!=READER_APP_ID:
        raise RuntimeError('reader deployment identity changed; refuse to stop another app')
    client=await _Client.from_env()
    await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app.app_id,source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT))



def recorded_activity(root,state):
    training=read_rows(root/'experiment/seed42/adapter/rollouts-rank0.jsonl')
    before=read_rows(root/'evaluations/base.jsonl')
    after=read_rows(root/'evaluations/rl.jsonl')
    sft=read_rows(root/'evaluations/sft.jsonl')
    episodes=training+before+sft+after
    turns=[t for ep in episodes for t in ep.get('turns',[])]
    history=[h for ep in episodes for h in ep.get('history',[]) if isinstance(h,dict)][-500:]
    def mean(key):
        values=[h[key] for h in history if isinstance(h.get(key),(int,float)) and math.isfinite(h[key])]
        return sum(values)/len(values) if values else None
    state['kernel_evals']=len(turns)
    state['recent_evaluation']={'calls':len(turns),'queue_wait_s':mean('queue_wait_s'),'eval_s':mean('eval_s')}
    state['activity']=state.get('phase','')+' · '+str(len(training))+' training episodes collected'
    protocol=read_json(root/'protocol.json') or read_json(root/'contract.json').get('protocol',{})
    state['sft_groups']={name:summarize([ep for ep in sft if ep.get('task_id') in ids])
                         for name,ids in protocol.get('evaluation_groups',protocol.get('groups',{})).items()}
    return state


def validation_attempts(root):
    attempts=[]
    for folder in sorted((root/'validation').glob('*'),key=lambda p:p.stat().st_mtime):
        if not folder.is_dir():continue
        result=read_json(folder/'result.json');cases=read_rows(folder/'cases.jsonl')
        failed=[r['case'] for r in cases if r.get('passed') is False]
        required_failed=any(r.get('passed') is False and r.get('required',True) for r in cases)
        attempts.append({'reservation_id':folder.name,'state':'passed' if result.get('passed') else
            'failed' if required_failed or result.get('passed') is False else 'incomplete',
            'passed_cases':sum(r.get('passed') is True for r in cases),'failed_cases':failed,
            'not_run':result.get('not_run',[]),'complete':bool(result) and result.get('completed',True),
            'source_digest':read_json(folder/'contract.json').get('source_digest'),
            'protocol_digest':read_json(folder/'contract.json').get('protocol_digest')})
    return attempts


def make_app(root,reload_volume,stop_reader=stop_owned_reader):
    root=Path(root);policy_root=root;app=FastAPI();lock=threading.Lock();valid={};last_reload=0.
    def closed():
        return lease_expired(read_json(root/'reader-policy.json'),time.time(),
                             (root/'comparison.json').exists() or (root/'TERMINAL.json').exists() or
                             (root.parent/'corrected-20261019/comparison.json').exists() or (root.parent/'corrected-20261019/TERMINAL.json').exists())
    @app.on_event('startup')
    async def start_lifecycle_guard():
        async def watch():
            while True:
                try:await asyncio.to_thread(reload_volume)
                except Exception:pass
                if closed():
                    try:
                        await stop_reader(read_json(root/'reader-policy.json'))
                        return
                    except Exception as error:
                        print('Reader shutdown retry:',type(error).__name__,flush=True)
                await asyncio.sleep(5)
        app.state.lifecycle_task=asyncio.create_task(watch())
    @app.on_event('shutdown')
    async def end_lifecycle_guard():
        app.state.lifecycle_task.cancel()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    opener=urllib.request.build_opener(NoRedirect)
    @app.middleware('http')
    async def authorize(request,call_next):
        if closed():return JSONResponse({'error':'Experiment reader lease ended'},status_code=410)
        token=request.headers.get('X-Kernel-Dashboard-Token','')
        if not token or len(token)>512:return JSONResponse({'error':'Forbidden'},status_code=403)
        key=hashlib.sha256(token.encode()).hexdigest()
        if valid.get(key,0)<time.monotonic():
            req=urllib.request.Request(EXISTING_READER+'/state?run=seed42',headers={'X-Kernel-Dashboard-Token':token})
            try:
                # Authentication remains with the unchanged old reader; token is neither logged nor saved.
                import asyncio
                def check():
                    with opener.open(req,timeout=8) as response:
                        if response.status!=200:raise ValueError('reader rejected request')
                await asyncio.to_thread(check)
            except Exception:return JSONResponse({'error':'Existing authentication reader unavailable or rejected request'},status_code=403)
            valid.clear();valid[key]=time.monotonic()+30
        return await call_next(request)
    def snapshot(run='better'):
        nonlocal last_reload
        if run not in ('better','corrected'):raise HTTPException(404)
        selected=policy_root if run=='better' else policy_root.parent/'corrected-20261019'
        with lock:
            if time.monotonic()-last_reload>15:reload_volume();last_reload=time.monotonic()
            state=state_from(selected/'experiment','seed42')
            status=read_json(selected/'status.json')
            if status:state['phase']=status.get('phase',state['phase']);state['campaign_status']=status
            budget=read_json(selected/'budget.json')
            state.update(cost=budget.get('new_actual_reported_usd'),budget_cap_usd=100,
                estimated_upper_usd=budget.get('new_reserved_upper_usd'),
                model='Qwen2.5-Coder-7B · verified SFT → GRPO · new protocol',
                hardware='H100-80GB · 3-turn training · separate evaluator',
                activity=status.get('detail',state['activity']),
                warning='New protocol and training shapes. Compare matched frozen/SFT/RL evaluations on original task shapes; training reward is not held-out improvement. Spend shown is a conservative reservation; actual billing is unavailable until reported.')
            state.update(goal=32,episodes_per_update=8,reader_lease=read_json(policy_root/'reader-policy.json'))
            state['budget_ledger']=budget
            state['run_id']=selected.name
            identity=read_json(selected/'contract.json')
            state['protocol_digest']=identity.get('protocol_digest')
            state['source_digest']=identity.get('source_digest')
            if run=='corrected':
                state['validation_attempts']=validation_attempts(selected)
                if not status:
                    latest=state['validation_attempts'][-1] if state['validation_attempts'] else None
                    state['phase']='Validation '+latest['state'] if latest else 'Validation pending'
                    if latest:
                        state['protocol_digest']=latest.get('protocol_digest')
                        state['source_digest']=latest.get('source_digest')
                state['history_label']='Corrected run; prior biased run is separate history'
                state['timing_scope']='Amortized graph-replay event interval with unsubtracted launch overhead'
                state['warning']='Amortized graph-replay event interval with unsubtracted launch overhead. Best validated quality within 3 attempts; gamma 1. Quality is 0.25 + clipped log2 of conservative speedup; sub-parity/parity has no speed bonus. Prior biased timings are incomparable. Only matched base/SFT/RL evaluation measures held-out change. Spend is a reservation; actual billing is unavailable until reported.'
            else:
                state['history_label']='Prior biased run; retained history, timings incomparable to corrected run'
            return recorded_activity(selected,state)
    @app.get('/state')
    def state(run='better'):return snapshot(run)
    @app.get('/metrics.json')
    def metrics(run='better'):return snapshot(run)
    @app.get('/rollouts.jsonl')
    def rollouts(run='better'):
        if run not in ('better','corrected'):raise HTTPException(404)
        selected=policy_root if run=='better' else policy_root.parent/'corrected-20261019'
        path=selected/'experiment/seed42/adapter/rollouts-rank0.jsonl'
        return FileResponse(path,media_type='application/x-ndjson',filename='better-rollouts.jsonl') if path.exists() else Response('',media_type='application/x-ndjson')
    return app

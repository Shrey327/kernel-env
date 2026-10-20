"""Read-only campaign API plus the dashboard UI it serves. Pages are public shells; data needs the token."""
import hmac
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from dashboard import view

UI=Path(__file__).parent/'dashboard_ui'
PAGES={'/':'index.html','/methodology':'methodology.html','/methodology.html':'methodology.html',
       '/journey':'journey.html','/journey.html':'journey.html','/pilot':'pilot/index.html','/pilot/':'pilot/index.html'}
PILOT={p.name for p in (UI/'pilot').glob('*')}  # fixed allowlist: request paths never reach the filesystem
COOKIE='kernel_dashboard'


def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return {} if default is None else default


def read_rows(path):
    if not Path(path).exists():return []
    rows=[]
    for line in Path(path).read_text().splitlines():
        try:rows.append(json.loads(line))
        except ValueError:pass
    return rows


def state_from(root,seed):
    folder=root/seed
    manifest=read_json(folder/'manifest.json') or read_json(root/'manifest.json') or {
        'model':'Qwen/Qwen2.5-Coder-7B-Instruct','steps':64,'episodes_per_update':32,
        'budget_cap_usd':100,'prior_reported_usd':7.24075504}
    checkpoints=sorted((folder/'adapter').glob('checkpoint-*/trainer_state.json'),
                       key=lambda p:int(p.parent.name.split('-')[-1]))
    trainer=read_json(folder/'adapter/live_trainer_state.json')
    if not trainer and checkpoints:trainer=read_json(checkpoints[-1])
    snapshot={'manifest':manifest,'before':read_rows(root/'before.jsonl'),
              'after':read_rows(folder/'after.jsonl'),'train':read_rows(folder/'adapter/rollouts-rank0.jsonl'),
              'trainer':trainer,'comparison':read_json(folder/'comparison.json')}
    log=(folder/'train.log').read_text()[-300000:] if (folder/'train.log').exists() else ''
    billing=read_json(root/'budget.json')
    state=view(snapshot,log,billing.get('reported_total_usd',manifest.get('prior_reported_usd')),episode_limit=64)
    status=read_json(root/'status.json') or read_json(root.parent/'status.json')
    if not snapshot['comparison']:state['phase']=status.get('phase','Preparing')
    if status.get('active_seed') and status['active_seed']!=int(seed.removeprefix('seed')) and not snapshot['train']:
        state['phase']='Waiting for this seed'
    state['seed']=seed
    state['campaign_status']=status
    state['model']=manifest.get('model','Qwen2.5-Coder-7B')+' · '+seed+' · bf16'
    state['fleet']={name:read_rows(root.parent/(name+'_utilization.jsonl'))[-120:] for name in ('trainer','evaluator')}
    env_events=read_rows(root.parent/'evals.jsonl')
    all_steps=[r for r in env_events if r.get('event')=='step']
    steps=all_steps[-500:]
    state['kernel_evals']=len(all_steps)
    state['recent_evaluation']={'calls':len(steps),'queue_wait_s':sum(r.get('queue_wait_s',0) for r in steps)/max(len(steps),1),
                                'eval_s':sum(r.get('eval_s',0) for r in steps)/max(len(steps),1)}
    state['episode_counts']={k:len(snapshot.get(k,[])) for k in ('before','train','after')}
    if manifest.get('created_at_unix'):
        started=manifest.get('deadline_unix',manifest['created_at_unix']+8*3600-120)-8*3600+120
        ended=status.get('updated_unix',time.time()) if status.get('phase','').startswith(('Complete','Stopped')) else time.time()
        state['estimated_upper_usd']=manifest.get('prior_reported_usd',7.24075504)+min(
            max(ended-started,0),8*3600)*9.7/3600
    mtimes=[p.stat().st_mtime for p in (folder/'adapter/live_trainer_state.json',root/'status.json') if p.exists()]
    state['updated']=max(mtimes+[r['timestamp'] for rows in state['fleet'].values() for r in rows],default=time.time())
    state['inference_benchmark']=read_json(root/'inference_benchmark.json')
    state['activity']=f"{state['phase']} · {state['episode_counts']['train']} training episodes collected"
    state['warning']='Latest 64 training episodes shown below; download the full rollout log for all source and feedback. Billing can lag; the resource estimate is an upper bound, not a charge.'
    return state


def make_app(root,token,reload_volume=lambda:None):
    if not token:raise ValueError('dashboard token is required')
    root=Path(root);app=FastAPI();lock=threading.Lock();last_reload=0.0

    @app.middleware('http')
    async def authenticate(request:Request,call_next):
        path=request.url.path
        public=path=='/health' or path in PAGES or (path.startswith('/pilot/') and path.endswith('.html'))
        supplied=request.headers.get('X-Kernel-Dashboard-Token') or request.cookies.get(COOKIE,'')
        if not public and not hmac.compare_digest(supplied.encode(),token.encode()):
            return JSONResponse({'error':'unauthorized'},status_code=401)
        return await call_next(request)

    def page(name):
        return FileResponse(UI/name,media_type='text/html; charset=utf-8',headers={'Cache-Control':'no-store',
            'X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin'})
    for route,name in PAGES.items():
        app.add_api_route(route,lambda name=name:page(name),methods=['GET'],include_in_schema=False)

    @app.get('/pilot/{name}')
    def pilot(name:str):
        if name not in PILOT:raise HTTPException(404)
        return page('pilot/'+name) if name.endswith('.html') else FileResponse(UI/'pilot'/name,headers={'Cache-Control':'no-store'})

    @app.get('/auth')
    def auth():
        # Reached only past the header check; browsers then send the HttpOnly cookie on pages, fetches and downloads.
        response=Response(status_code=204)
        response.set_cookie(COOKIE,token,max_age=7*86400,httponly=True,secure=True,samesite='strict')
        return response

    def seed_path(run):
        nonlocal last_reload
        if run not in ('seed42','seed123'):raise HTTPException(404)
        with lock:
            if time.monotonic()-last_reload>20:
                reload_volume();last_reload=time.monotonic()
        return root/run

    @app.get('/health')
    def health():return {'status':'ok'}

    @app.get('/state')
    def state(run='seed42'):
        seed_path(run)
        return JSONResponse(state_from(root,run),headers={'Cache-Control':'no-store'})

    @app.get('/metrics.json')
    def metrics(run='seed42'):
        seed_path(run)
        return JSONResponse(state_from(root,run)['metrics'],headers={'Cache-Control':'no-store'})

    @app.get('/rollouts.jsonl')
    def rollouts(run='seed42'):
        path=seed_path(run)/'adapter/rollouts-rank0.jsonl'
        if not path.is_file():raise HTTPException(404,detail='No rollouts collected yet')
        return FileResponse(path,media_type='application/x-ndjson',filename=run+'-rollouts.jsonl')

    return app

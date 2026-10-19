"""Bounded stages for a new protocol; never reuse or overwrite the failed v5 run."""
import json
import os
import signal
import sys
import subprocess
import threading
import time
from pathlib import Path

import modal

if modal.is_local():
    from modal_app import train_image
else:
    # Remote workers need only the registered function body; avoid importing the local launcher module.
    train_image=modal.Image.debian_slim(python_version='3.12')

sys.path.insert(0,str(Path(__file__).parent/'train'))

app=modal.App('kernel-env-better-20261019')
runs=modal.Volume.from_name('kernel-env-runs')


def mark_started(reservation):
    path=Path('/runs/better-20261019/started')/(reservation['reservation_id']+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'body_started_unix':time.time(),'source_digest':reservation['source_digest']}))
    runs.commit()


def verify_source(reservation):
    import hashlib
    root=Path('/root')
    for name,expected in reservation.get('source_files',{}).items():
        if name.startswith(('kernel_env/','train/')) or name=='modal_better.py':
            actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
            if actual!=expected:raise RuntimeError('packaged source digest mismatch: '+name)


@app.function(image=train_image,gpu='A100-80GB',cpu=(4,8),memory=(32768,65536),
              timeout=900,startup_timeout=180,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/runs':runs})
def validate(deadline_unix,reservation):
    verify_source(reservation)
    mark_started(reservation)
    if deadline_unix<=time.time() or deadline_unix>time.time()+1200:
        raise ValueError('validation deadline is outside the reserved interval')
    root=Path('/runs/better-20261019/validation')/reservation['reservation_id']
    root.mkdir(parents=True,exist_ok=False)
    (root/'validation-reservation.json').write_text(json.dumps(reservation,indent=2))
    started=time.time()
    cmd=['python','-m','pytest','tests/test_worker_gpu.py','tests/test_gpu_fingerprint.py','-q','-s','-k',
         'good_kernel_is_correct or wrong_output or aliasing_input or nondefault_stream or identity_copy or gpu_fingerprint']
    with (root/'validation.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd='/root/kernel_env',stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,env={**os.environ,'PYTHONPATH':'/root:/root/kernel_env/tests'})
        try:rc=proc.wait(timeout=min(840,max(1,deadline_unix-time.time()-20)))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rc=124
    result={'returncode':rc,'seconds':time.time()-started,'finished_unix':time.time(),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),'new_budget_cap_usd':100,'reservation_id':reservation['reservation_id']}
    (root/'validation-result.json').write_text(json.dumps(result,indent=2));runs.commit()
    if rc:raise RuntimeError(f'validation failed ({rc}); inspect better-20261019/validation.log')
    selection=root.parents[1]/'selection.json'
    selection.write_text(json.dumps({'validation':str(root),'validation_gpu':result['gpu']}));runs.commit()
    return result

# CPU-only reader; existing Sites token is verified by the unchanged original reader.
dashboard_app=modal.App('kernel-env-better-dashboard')
dashboard_image=(modal.Image.debian_slim(python_version='3.12').pip_install('openenv-core[core]==0.3.0')
    .env({'PYTHONPATH':'/root:/root/train'})
    .add_local_dir(Path(__file__).parent/'kernel_env','/root/kernel_env',ignore=['.venv','**/__pycache__','uv.lock'])
    .add_local_dir(Path(__file__).parent/'train','/root/train',ignore=['.venv','**/__pycache__','uv.lock']))

@dashboard_app.function(image=dashboard_image,cpu=(0.125,0.25),memory=(512,512),timeout=45,min_containers=1,max_containers=1,
                       scaledown_window=15,volumes={'/runs':runs.read_only()})
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def better_reader():
    from better_dashboard import make_app
    return make_app(Path('/runs/better-20261019'),runs.reload)

@app.function(image=train_image,gpu='H100',cpu=(4,8),memory=(32768,65536),timeout=900,
              startup_timeout=180,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':runs})
def validate_h100(deadline_unix,reservation):
    # Same verifier and source as A100 validation; only capacity choice changes.
    return validate.local(deadline_unix,reservation)

hf_cache=modal.Volume.from_name('kernel-env-hf-cache')


def execute_stage(phase,deadline,reservation,budget):
    verify_source(reservation)
    mark_started(reservation)
    os.environ['KERNELBENCH_DIR']='/opt/KernelBench/KernelBench'
    root=Path('/runs/better-20261019');selection=json.loads((root/'selection.json').read_text())
    if deadline<=time.time()+60:raise TimeoutError('no reserved stage time remains')
    validation=json.loads((Path(selection['validation'])/'validation-result.json').read_text())
    if validation['returncode']!=0:raise RuntimeError('GPU validation gate failed')
    if 'H100' not in validation['gpu']:raise RuntimeError('new experiment requires the validated H100 protocol')
    out=root/phase/reservation['reservation_id'];out.parent.mkdir(parents=True,exist_ok=True)
    (root/'budget.json').write_text(json.dumps(budget,indent=2))
    status_path=root/'status.json'
    def status(label,**extra):
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps({'phase':label,'updated_unix':time.time(),'detail':phase,**extra}));temp.replace(status_path)
    status({'warmup':'Verified SFT warmup','pilot':'Utilization pilot','train':'Training'}.get(phase,'Frozen evaluation'))
    stop=threading.Event()
    def commit():
        while not stop.wait(15):
            try:
                progress=out/'progress.json'
                if progress.exists():
                    info=json.loads(progress.read_text());status(info.get('stage',phase),progress=info,heartbeat_unix=time.time())
                runs.commit()
            except Exception:pass
    threading.Thread(target=commit,daemon=True).start()
    server=None;proc=None
    try:
        if phase=='warmup':
            from runtime_support import monitor
            threading.Thread(target=monitor,args=(root/'trainer_utilization.jsonl',stop,{0}),daemon=True).start()
            command=['python','-u','better_warmup.py','--out',str(out),'--deadline',str(deadline-30)]
        else:
            from runtime_support import _start_env_server
            from better_protocol import training_copy
            from kernel_env.server.tasks import load_tasks
            if phase in ('pilot','train'):
                shaped=training_copy(load_tasks(),Path('/tmp')/('kernelbench-training-'+reservation['reservation_id']))
                os.environ['KERNELBENCH_DIR']=str(shaped)
            server,eval_log=_start_env_server('1','127.0.0.1','paired_speedup',1,False,90,3)
            os.environ['KERNEL_ENV_URL']='http://127.0.0.1:8000'
            experiment=root/'experiment';(experiment/'seed42').mkdir(parents=True,exist_ok=True)
            if not (experiment/'manifest.json').exists():
                from better_protocol import CURRICULUM
                manifest={'model':'Qwen/Qwen2.5-Coder-7B-Instruct','steps':32,'episodes_per_update':8,'max_turns':3,
                   'budget_cap_usd':100,'prior_reported_usd':None,'hardware':'H100 trainer + H100 evaluator',
                   'candidate_tasks':CURRICULUM,'protocol':'verified SFT; fresh paired timings; gamma0.4; trajectory LOO; no partial credit',
                   'training_shapes':'two warmup variants, fixed medium RL; evaluation original shapes'}
                (experiment/'manifest.json').write_text(json.dumps(manifest,indent=2))
            command=['python','-u','better_campaign.py','--root',str(root),'--phase',phase if phase in ('pilot','train') else 'evaluate',
                     '--out',str(out),'--deadline',str(deadline-45)]
            if phase=='pilot' and selection.get('reusable_control'):
                control=selection['reusable_control']
                if control['warmup']!=selection['warmup']:raise RuntimeError('control SFT artifact changed')
                command += ['--reuse-control',control['path']]
            if phase.startswith('eval-'):command+=['--arm',phase.removeprefix('eval-')]
            from runtime_support import monitor
            for name,index in [('trainer',0),('evaluator',1)]:
                threading.Thread(target=monitor,args=(root/(name+'_utilization.jsonl'),stop,{index}),daemon=True).start()
        # Logs live outside the script-owned output folder, which uses exist_ok=False.
        log_path=out.parent/(out.name+'.log')
        with log_path.open('w') as log:
            proc=subprocess.Popen(command,cwd='/root/train',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
               env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','TOKENIZERS_PARALLELISM':'false','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
            try:rc=proc.wait(timeout=max(1,deadline-time.time()-20))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError('reserved stage deadline')
        if rc:raise RuntimeError(f'{phase} exited {rc}: {log_path}')
        result=json.loads((out/'result.json').read_text())
        if phase=='warmup':
            selection['warmup']=str(out);(root/'selection.json').write_text(json.dumps(selection,indent=2))
        status('Complete' if (root/'comparison.json').exists() else phase+' complete',result=result)
        return {'phase':phase,'out':str(out),'result':result}
    except BaseException as error:
        status('Stopped: '+phase,error=str(error)[:1000]);raise
    finally:
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        if server:
            server.terminate()
            try:server.wait(timeout=5)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        stop.set();runs.commit()


@app.function(image=train_image,gpu='H100',cpu=(4,8),memory=(32768,65536),timeout=1200,
              startup_timeout=180,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/runs':runs,'/root/.cache/huggingface':hf_cache})
def warmup(deadline,reservation,budget):
    return execute_stage('warmup',deadline,reservation,budget)


@app.function(image=train_image,gpu='H100!:2',cpu=(4,8),memory=(32768,65536),timeout=5400,
              startup_timeout=180,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/runs':runs,'/root/.cache/huggingface':hf_cache})
def experiment_stage(phase,deadline,reservation,budget):
    return execute_stage(phase,deadline,reservation,budget)


@app.function(image=train_image,cpu=1,memory=2048,timeout=90,startup_timeout=120,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':runs})
def import_smoke(deadline,reservation):
    if time.time()>deadline:raise TimeoutError('smoke deadline')
    verify_source(reservation)
    mark_started(reservation)
    import runtime_support,better_protocol,better_campaign,better_warmup
    from kernel_env.server.validation_memory import input_fingerprint
    from kernel_env.server.benchmark_protocol import cuda_coverage
    import importlib.metadata
    return {'passed':True,'source_digest':reservation['source_digest'],
            'versions':{name:importlib.metadata.version(name) for name in ('torch','triton','trl','peft')}}

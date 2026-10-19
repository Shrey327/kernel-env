"""Distinct corrected experiment; only bounded validation is enabled before review."""
import json,os,signal,subprocess,sys,time,threading
from pathlib import Path
import modal
sys.path.insert(0,str(Path(__file__).parent/'train'))
if modal.is_local():
    from modal_app import train_image
else:train_image=modal.Image.debian_slim(python_version='3.12')
from corrected_contract import RUN,verify_sources
app=modal.App('kernel-env-corrected-20261019')
runs=modal.Volume.from_name('kernel-env-runs')


@app.function(image=train_image,gpu='H100',cpu=(4,8),memory=(32768,65536),timeout=2940,
              startup_timeout=90,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':runs})
def validate(reservation,contract,budget):
    verify_sources(Path(__file__).parent,contract['source_files'])
    os.environ.update(contract['protocol']['env'])  # hybrid contract + few-shot prompt, identical for every phase
    root=Path('/runs')/RUN;out=root/'validation'/reservation['reservation_id'];out.mkdir(parents=True,exist_ok=False)
    (root/'progress').mkdir(exist_ok=True)
    progress=root/'progress'/(reservation['reservation_id']+'.json')
    progress.write_text(json.dumps({'semantic_progress':'body started','updated_unix':time.time()}))
    (out/'contract.json').write_text(json.dumps(contract,indent=2));(root/'budget.json').write_text(json.dumps(budget,indent=2));runs.commit()
    stop=threading.Event()
    def commits():
        while not stop.wait(15):
            try:runs.commit()
            except Exception:pass
    threading.Thread(target=commits,daemon=True).start()
    log=out/'preflight.log';proc=None
    try:
        with log.open('w') as f:
            proc=subprocess.Popen(['python','-u','corrected_preflight.py','--out',str(out),'--contract',str(out/'contract.json'),
                                   '--progress',str(progress),'--deadline',str(reservation['deadline_unix']-50)],cwd='/root/train',
                                   stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=max(1,reservation['deadline_unix']-time.time()-45))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('remote absolute validation deadline')
        if rc:raise RuntimeError('preflight failed; see '+str(log))
        return json.loads((out/'result.json').read_text())
    finally:
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        stop.set();runs.commit()


@app.function(image=train_image,gpu='H100:2',cpu=(4,8),memory=(32768,65536),timeout=9000,
              startup_timeout=90,retries=0,max_containers=1,scaledown_window=2,volumes={'/runs':runs})
def stage(phase,reservation,contract,budget,gates):
    from corrected_contract import require_gate
    verify_sources(Path(__file__).parent,contract['source_files'])
    for name in ('gpu-validation','reward-calibration','sft-verification'):require_gate(gates[name],contract,name)
    os.environ.update(contract['protocol']['env'])
    if phase=='train':
        require_gate(gates['pilot'],contract,'pilot');require_gate(gates['independent-review'],contract,'independent-review')
    if phase.startswith('eval-'):require_gate(gates['independent-review'],contract,'independent-review')
    root=Path('/runs')/RUN;root.mkdir(exist_ok=True);out=root/phase/reservation['reservation_id'];out.parent.mkdir(exist_ok=True)
    (root/'progress').mkdir(exist_ok=True);progress=root/'progress'/(reservation['reservation_id']+'.json')
    manifest=root/'contract.json'
    if manifest.exists() and json.loads(manifest.read_text())!=contract:raise ValueError('preserve frozen corrected contract')
    manifest.write_text(json.dumps(contract,indent=2));(root/'gates.json').write_text(json.dumps(gates,indent=2));(root/'budget.json').write_text(json.dumps(budget,indent=2))
    status=root/'status.json';stop=threading.Event();server=None;proc=None;eval_log=None
    def update():
        files=[root/'experiment/seed42/adapter/live_trainer_state.json',out/'adapter/live_trainer_state.json',out/'train.log',out/'evaluate.log']
        if eval_log:files.append(Path(eval_log))
        signature=[(str(p),p.stat().st_size) for p in files if p.exists()]
        progress.write_text(json.dumps({'semantic_progress':signature or ['body started'],'updated_unix':time.time()}))
        status.write_text(json.dumps({'phase':'Training' if phase=='train' else 'Utilization pilot' if phase=='pilot' else 'Final evaluation','detail':phase,'updated_unix':time.time(),'run_id':RUN,'source_digest':contract['source_digest'],'protocol_digest':contract['protocol_digest']}))
    def commit():
        while not stop.wait(15):
            try:update();runs.commit()
            except Exception:pass
    update();runs.commit();threading.Thread(target=commit,daemon=True).start()
    try:
        os.environ['KERNELBENCH_DIR']='/opt/KernelBench/KernelBench'
        if phase in ('pilot','train'):
            from kernel_env.server.tasks import load_tasks
            from better_protocol import training_copy
            os.environ['KERNELBENCH_DIR']=str(training_copy(load_tasks(),Path('/tmp')/('corrected-training-'+reservation['reservation_id']),variant=contract['protocol']['training_variant']))
        from runtime_support import _start_env_server,monitor
        server,eval_log=_start_env_server('1','127.0.0.1','calibrated_speedup_v2',1,False,600 if phase.startswith('eval-') else 240,3)
        for label,index in [('trainer',0),('evaluator',1)]:threading.Thread(target=monitor,args=(root/(label+'_utilization.jsonl'),stop,{index}),daemon=True).start()
        os.environ['KERNEL_ENV_URL']='http://127.0.0.1:8000'
        with (out.parent/(out.name+'.log')).open('w') as f:
            proc=subprocess.Popen(['python','-u','corrected_campaign.py','--root',str(root),'--out',str(out),'--contract',str(manifest),
                  '--phase',phase,'--deadline',str(reservation['deadline_unix']-60)],cwd='/root/train',stdout=f,stderr=subprocess.STDOUT,
                  start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','TOKENIZERS_PARALLELISM':'false','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
            try:rc=proc.wait(timeout=max(1,reservation['deadline_unix']-time.time()-45))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('remote absolute stage cutoff')
        if rc:raise RuntimeError('corrected stage failed; preserve logs, no automatic retry')
        result=json.loads((out/'result.json').read_text());result.update({k:contract[k] for k in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')});result['stage']=phase
        status.write_text(json.dumps({'phase':('Complete' if (root/'comparison.json').exists() else phase+' complete') if result.get('passed') else 'Stopped: '+phase+' gate failed','updated_unix':time.time(),'result':result}))
        return result
    except BaseException as error:
        status.write_text(json.dumps({'phase':'Stopped: '+phase,'error':str(error)[:500],'updated_unix':time.time()}));raise
    finally:
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if server:
            server.terminate()
            try:server.wait(timeout=5)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        stop.set();runs.commit()

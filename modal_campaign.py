"""Bounded GPU campaign and a separately authenticated, CPU-only dashboard reader."""
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

ROOT=Path(__file__).parent
if modal.is_local():
    from modal_app import env_image, train_image
    env_image=(env_image.add_local_file(ROOT/'modal_app.py','/root/modal_app.py')
        .add_local_dir(ROOT/'train','/root/train',ignore=['.venv','**/__pycache__']))
    train_image=train_image.add_local_file(ROOT/'modal_app.py','/root/modal_app.py')
else:
    env_image=train_image=None
runs=modal.Volume.from_name('kernel-env-runs',create_if_missing=True)
hf_cache=modal.Volume.from_name('kernel-env-hf-cache',create_if_missing=True)
RUN='larger-20261019-v5-noise-aware'
HOURS=8
app=modal.App('kernel-env-larger-20261019')
dashboard_app=modal.App('kernel-env-live-dashboard')


def monitor(path,stop,indices):
    while not stop.is_set():
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw',
                '--format=csv,noheader,nounits'],text=True,timeout=4)
            with Path(path).open('a') as f:
                for row in text.strip().splitlines():
                    i,name,util,mem_util,mem,power=[v.strip() for v in row.split(',')]
                    if int(i) not in indices:continue
                    f.write(json.dumps({'timestamp':time.time(),'index':int(i),'name':name,
                        'gpu_util_pct':float(util),'memory_util_pct':float(mem_util),
                        'memory_used_mib':float(mem),'power_w':float(power)})+'\n')
        except Exception as error:print('utilization sample unavailable:',type(error).__name__,flush=True)
        stop.wait(5)


@app.function(image=train_image,gpu='H100!:2',cpu=(4,8),memory=(32768,65536),
              timeout=HOURS*3600+120,startup_timeout=600,retries=0,
              volumes={'/runs':runs,'/root/.cache/huggingface':hf_cache},max_containers=1,scaledown_window=2)
def train_campaign(deadline):
    if not time.time()+600 < deadline <= time.time()+HOURS*3600:
        raise ValueError('deadline must leave ten minutes and be within eight hours of launch')
    root=Path('/runs')/RUN
    if root.exists():raise RuntimeError('preserve existing campaign; choose a new run name for another attempt')
    root.mkdir()
    def status(phase, **extra):
        target=root/'status.json'
        temp=root/'status.json.tmp'
        temp.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),**extra}))
        temp.replace(target)
    status('GPU checks')
    runs.commit()
    stop=threading.Event()
    threading.Thread(target=monitor,args=(root/'trainer_utilization.jsonl',stop,{0}),daemon=True).start()

    def commit_loop():
        while not stop.wait(30):
            try:runs.commit()
            except Exception as error:print('volume checkpoint warning:',type(error).__name__,flush=True)
    threading.Thread(target=commit_loop,daemon=True).start()
    threading.Thread(target=monitor,args=(root/'evaluator_utilization.jsonl',stop,{1}),daemon=True).start()
    proc=None;server=None
    try:
        from modal_app import _start_env_server
        result=subprocess.run(['python','-m','pytest','tests','-q'],cwd='/root/kernel_env',
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},timeout=600,capture_output=True,text=True)
        (root/'gpu_preflight.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('same-image GPU preflight failed; inspect gpu_preflight.log')
        status('Environment connection check')
        server,log_path=_start_env_server('1','127.0.0.1','roofline',2,False,60,2)
        (root/'evals.jsonl').symlink_to(log_path)
        from kernel_env import KernelEnv,KernelAction
        from kernel_env.tests.kernels import GOOD
        with KernelEnv(base_url='http://127.0.0.1:8000').sync() as smoke:
            smoke.reset(task_id='level1/19_ReLU',split='train',seed=991)
            result=smoke.step(KernelAction(code=GOOD))
            if not result.observation.correct:raise RuntimeError('real-GPU localhost handshake/execution failed')
            (root/'connection_smoke.json').write_text(json.dumps({'correct':True,'stage':result.observation.stage,'reward':result.reward}))
        url='http://127.0.0.1:8000'
        status('Preparing')
        cmd=['python','-u','campaign.py','--out',str(root/'experiment'),'--deadline',str(deadline-120),'--steps','64']
        # campaign.py owns the experiment subdirectory; shared telemetry stays in its parent.
        with (root/'campaign.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd='/root/train',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','KERNEL_ENV_URL':url,'TOKENIZERS_PARALLELISM':'false','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
            try:rc=proc.wait(timeout=max(deadline-time.time()-30,1))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError('campaign resource deadline')
        if rc:raise RuntimeError(f'campaign exited {rc}; inspect campaign.log')
    except BaseException as error:
        status('Stopped with incomplete work',error=repr(error))
        raise
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
        if server:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        stop.set();runs.commit()


dashboard_image=(modal.Image.debian_slim(python_version='3.12').pip_install('openenv-core[core]==0.3.0')
    .env({'PYTHONPATH':'/root:/root/train'})
    .add_local_dir(ROOT/'kernel_env','/root/kernel_env',ignore=['.venv','**/__pycache__','uv.lock'])
    .add_local_dir(ROOT/'train','/root/train',ignore=['.venv','**/__pycache__'])) if modal.is_local() else None


@dashboard_app.function(image=dashboard_image,cpu=0.125,memory=1024,timeout=60,max_containers=1,
              scaledown_window=15,volumes={'/runs':runs.read_only()},
              secrets=[modal.Secret.from_dict({'KERNEL_DASHBOARD_TOKEN':os.environ.get('KERNEL_DASHBOARD_TOKEN','')})])
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def live_dashboard():
    from campaign_dashboard import make_app
    return make_app(Path('/runs')/RUN/'experiment',os.environ['KERNEL_DASHBOARD_TOKEN'],runs.reload)

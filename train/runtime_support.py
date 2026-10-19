import os,subprocess,time,json,urllib.request
from pathlib import Path

def _start_env_server(gpu_ids: str, host: str, reward: str, workers_per_gpu: int, partial_credit: bool,
                      eval_timeout: float = 300.0, max_turns: int = 4):
    """uvicorn env server on `gpu_ids`; returns (process, eval log path) once /health answers."""
    eval_log = f"/runs/evals-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"  # one JSON line per env reset/step
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "kernel_env.server.app:app", "--host", host, "--port", "8000"],
        env={**os.environ, "KERNEL_ENV_GPUS": gpu_ids, "KERNEL_ENV_MAX_SESSIONS": "256",
             "KERNEL_ENV_EVAL_LOG": eval_log, "KERNEL_ENV_REWARD": reward,
             "KERNEL_ENV_WORKERS_PER_GPU": str(workers_per_gpu),
             "KERNEL_ENV_PARTIAL_CREDIT": "1" if partial_credit else "0",
             "KERNEL_ENV_MAX_TURNS": str(max_turns),
             "KERNEL_ENV_EVAL_TIMEOUT": str(eval_timeout)},
    )
    local = "http://[::1]:8000" if ":" in host else "http://127.0.0.1:8000"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{local}/health", timeout=1)
            return proc, eval_log
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(1)
    proc.kill()
    raise RuntimeError("env server did not become healthy")

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


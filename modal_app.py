"""Run kernel_env and train/ on Modal GPUs.

    modal run modal_app.py::gpu_tests                     # GPU worker tests on the eval GPU type
    modal deploy modal_app.py                             # env server at https://<ws>--kernel-env-serve.modal.run
    modal run modal_app.py::job --cmd "python baseline_eval.py --env-url \\$KERNEL_ENV_URL ..."
    modal run modal_app.py::train --cmd "accelerate launch ... grpo_kernel.py --env_url \\$KERNEL_ENV_URL ..."

GPU types are read at launch: KERNEL_ENV_GPU (gpu_tests/serve, default L40S), TRAIN_GPU (job, default
H100:2), TRAINER_GPU / ENV_SERVER_GPU (train, default H100!:6 / H100!:2).
`job` runs the env server inside its own container on the last `eval_gpus` GPUs and the command on
the rest. `train` puts the env server in a separate container (private i6pn network, same region), so a
candidate kernel that faults its GPU cannot take the trainer down with it.
"""

import os
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).parent
KERNELBENCH_REF = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
EVAL_GPU = os.environ.get("KERNEL_ENV_GPU", "L40S")
TRAIN_GPU = os.environ.get("TRAIN_GPU", "H100:2")
TRAINER_GPU = os.environ.get("TRAINER_GPU", "H100!:6")
ENV_SERVER_GPU = os.environ.get("ENV_SERVER_GPU", "H100!:2")
REGION = os.environ.get("MODAL_REGION", "us-east")  # i6pn only connects containers within one region
IGNORE = [".venv", "**/__pycache__", "**/*.egg-info", "**/.pytest_cache", "uv.lock"]

# devel image: nvcc for language="cuda" (load_inline); torch wheels bring their own CUDA runtime
base = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl")
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench /opt/KernelBench",
        f"git -C /opt/KernelBench checkout {KERNELBENCH_REF}",
    )
    .env({"KERNELBENCH_DIR": "/opt/KernelBench/KernelBench", "PYTHONPATH": "/root"})
)
env_image = base.pip_install("torch==2.7.1", "openenv-core[core]>=0.2.1", "pytest", "ninja").add_local_dir(
    ROOT / "kernel_env", "/root/kernel_env", ignore=IGNORE
)
train_image = (
    base.pip_install("trl[vllm]==1.16.0", "peft", "datasets", "openenv-core[core]>=0.2.1", "pytest", "ninja")
    .pip_install("wandb")
    .add_local_dir(ROOT / "kernel_env", "/root/kernel_env", ignore=IGNORE)
    .add_local_dir(ROOT / "train", "/root/train", ignore=IGNORE)
)

app = modal.App("kernel-env")
runs = modal.Volume.from_name("kernel-env-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("kernel-env-hf-cache", create_if_missing=True)
# e.g. TRAIN_SECRETS=wandb,huggingface (Modal secret names providing WANDB_API_KEY / HF_TOKEN)
secrets = [modal.Secret.from_name(n) for n in os.environ.get("TRAIN_SECRETS", "").split(",") if n]
rendezvous = modal.Dict.from_name("kernel-env-rendezvous", create_if_missing=True)  # run_id -> env server URL


def _gpu_count() -> int:
    return len(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines())


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


@app.function(image=env_image, gpu=EVAL_GPU, timeout=1800)
def gpu_tests(args: str = "-q"):
    rc = subprocess.run(["python", "-m", "pytest", "tests", *args.split()], cwd="/root/kernel_env").returncode
    if rc:
        raise SystemExit(rc)


@app.function(
    image=env_image,
    gpu=EVAL_GPU,
    timeout=3600,
    scaledown_window=300,
    max_containers=int(os.environ.get("KERNEL_ENV_MAX_CONTAINERS", "4")),
    restrict_modal_access=True,  # submitted kernels run in this container
)
# one container = one eval GPU; autoscale at ~8 sessions per GPU, burst to 64
@modal.concurrent(max_inputs=64, target_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)  # executes agent code: never expose unauthenticated
def serve():
    from kernel_env.server.app import app as web

    return web


@app.function(
    image=train_image,
    gpu=TRAIN_GPU,
    timeout=24 * 3600,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
    secrets=secrets,
)
def job(cmd: str, eval_gpus: int = 1, reward: str = "speedup", workers_per_gpu: int = 2, partial_credit: bool = False,
        max_turns: int = 4):
    """Run `cmd` in /root/train with an env server on the last `eval_gpus` GPUs ($KERNEL_ENV_URL).

    reward: "speedup" or "roofline" (KERNEL_ENV_REWARD, see kernel_env/server/rubrics.py).
    workers_per_gpu: evals sharing one GPU for compile + correctness; benchmarks still run alone.
    """
    n = _gpu_count()
    if not 0 < eval_gpus < n:
        raise ValueError(f"need 0 < eval_gpus < {n} GPUs; set TRAIN_GPU=<type>:<count>")
    eval_ids = ",".join(map(str, range(n - eval_gpus, n)))
    train_ids = ",".join(map(str, range(n - eval_gpus)))
    server, eval_log = _start_env_server(eval_ids, "127.0.0.1", reward, workers_per_gpu, partial_credit,
                                       max_turns=max_turns)
    try:
        print(f"env server on GPU(s) {eval_ids} (log {eval_log}); running on GPU(s) {train_ids}: {cmd}", flush=True)
        rc = subprocess.run(
            cmd, shell=True, cwd="/root/train",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": train_ids, "KERNEL_ENV_URL": "http://127.0.0.1:8000"},
        ).returncode
    finally:
        server.terminate()
        runs.commit()
    if rc:
        raise SystemExit(rc)


@app.function(image=env_image, gpu=ENV_SERVER_GPU, timeout=24 * 3600, i6pn=True, region=REGION,
              volumes={"/runs": runs})
def env_server(run_id: str, reward: str = "speedup", workers_per_gpu: int = 2, partial_credit: bool = False,
               eval_timeout: float = 300.0):
    """Env server on all of this container's GPUs, reachable over i6pn; runs until `train` marks run_id done."""
    proc, eval_log = _start_env_server(",".join(map(str, range(_gpu_count()))), "::", reward, workers_per_gpu,
                                       partial_credit, eval_timeout)
    ip = socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]
    rendezvous[run_id] = f"http://[{ip}]:8000"
    print(f"env server for {run_id} at {rendezvous[run_id]} (log {eval_log})", flush=True)
    try:
        while proc.poll() is None and not rendezvous.get(f"{run_id}:done", False):
            time.sleep(30)
            runs.commit()  # eval log visible while the run is going
    finally:
        proc.terminate()
        runs.commit()


@app.function(image=train_image, gpu=TRAINER_GPU, timeout=24 * 3600, i6pn=True, region=REGION,
              retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.0),
              volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def train(cmd: str, reward: str = "roofline", workers_per_gpu: int = 2, partial_credit: bool = True,
          eval_timeout: float = 300.0):
    """Run `cmd` (GRPO) on all GPUs here, with the env server in its own container ($KERNEL_ENV_URL).

    Modal GPU containers can be preempted (nonpreemptible is not supported for GPU workloads). Retried up
    to twice on failure; grpo_kernel.py resumes from the newest checkpoint in its output_dir,
    which is committed to the volume every 2 minutes.
    """
    run_id = uuid.uuid4().hex[:12]
    call = env_server.spawn(run_id, reward, workers_per_gpu, partial_credit, eval_timeout)
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(120):
            runs.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    try:
        deadline = time.monotonic() + 3600  # the env container may queue for GPUs
        while (url := rendezvous.get(run_id)) is None:
            if time.monotonic() > deadline:
                raise RuntimeError("env server container did not start within 1 h")
            time.sleep(10)
        urllib.request.urlopen(f"{url}/health", timeout=10)
        print(f"run {run_id}: env server at {url}; training on {_gpu_count()} GPU(s): {cmd}", flush=True)
        rc = subprocess.run(cmd, shell=True, cwd="/root/train", env={**os.environ, "KERNEL_ENV_URL": url}).returncode
    finally:
        stop.set()
        rendezvous[f"{run_id}:done"] = True
        call.cancel()
        runs.commit()
        rendezvous.pop(run_id, None)
    if rc:
        raise SystemExit(rc)


@app.function(image=train_image, cpu=8, memory=65536, timeout=2 * 3600,
              volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def run_cpu(cmd: str):
    """Run a CPU-only command in /root/train (e.g. merge_lora.py) with the runs volume mounted."""
    rc = subprocess.run(cmd, shell=True, cwd="/root/train").returncode
    runs.commit()
    if rc:
        raise SystemExit(rc)

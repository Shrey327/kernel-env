"""Submission evaluation: static checks in-process, then worker.py in a subprocess on a leased GPU."""

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .roofline import describe, roofline
from .static_checks import check_source

WORKER = Path(__file__).with_name("worker.py")
# t_new below this fraction of the roofline floor is physically impossible -> rejected as a timing hack
ROOFLINE_REJECT = float(os.environ.get("KERNEL_ENV_ROOFLINE_REJECT", "0.8"))


def apply_roofline(res: Dict) -> Dict:
    timed = res.get("stage") == "timed"
    r = roofline(res.get("flops"), res.get("bytes"), res.get("dtype"), res.get("gpu"),
                 res.get("t_ref_ms"), res.get("t_new_ms") if timed else None)
    if not r:
        return res
    res = {**res, **r}
    if timed and res["t_new_ms"] < ROOFLINE_REJECT * r["t_min_ms"]:
        res.update(
            stage="correctness", correct=False, roofline_violation=True,
            feedback=f"Rejected: measured {res['t_new_ms']:.4f} ms is below the hardware floor of {r['t_min_ms']:.4f} ms "
                     f"for this much work on {res['gpu']}. No real kernel can do that; outputs were reused, work was "
                     f"skipped, or launches were not synchronized.",
        )
    elif timed:
        res["feedback"] = f"{res.get('feedback', '')}\n{describe(r, res['gpu'])}".strip()
    return res


REF_KEYS = ("flops", "bytes", "dtype")


class Evaluator:
    """Thread-safe; one instance is shared by all sessions so the GPU pool is global.

    workers_per_gpu > 1 lets evals share a GPU for compile + correctness (CPU-heavy JIT overlaps
    other work); benchmarks still run alone on the GPU via the worker's exclusive flock. The
    reference timing + roofline counts are measured once per task and reused (assumes one GPU type
    per evaluator).
    """

    def __init__(
        self,
        gpus: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        warmup_ms: int = 25,
        rep_ms: int = 100,
        workers_per_gpu: Optional[int] = None,
    ):
        self.workers_per_gpu = workers_per_gpu or int(os.environ.get("KERNEL_ENV_WORKERS_PER_GPU", "1"))
        self._gpus: "queue.Queue[str]" = queue.Queue()
        for g in gpus if gpus is not None else os.environ.get("KERNEL_ENV_GPUS", "0").split(","):
            for _ in range(self.workers_per_gpu):
                self._gpus.put(g.strip())
        self._ref_cache: Dict[str, Dict] = {}
        self._ref_lock = threading.Lock()
        # a hung kernel holds its GPU slot (and, in lockstep GRPO rollouts, the whole step) until this fires
        self.timeout_s = timeout_s or float(os.environ.get("KERNEL_ENV_EVAL_TIMEOUT", "300"))
        self.atol, self.rtol = atol, rtol
        self.warmup_ms, self.rep_ms = warmup_ms, rep_ms

    def evaluate(self, ref_src: str, code: str, language: str, seeds: List[int], task_id: str = "") -> Dict:
        """Result dict also carries gpu_id, queue_wait_s (waiting for a free GPU) and eval_s."""
        t0 = time.monotonic()
        errors = check_source(code, language, task_id)
        if errors:
            return {"stage": "format", "feedback": "Rejected before running:\n- " + "\n- ".join(errors[:10]),
                    "gpu_id": None, "queue_wait_s": 0.0, "eval_s": time.monotonic() - t0}
        key = hashlib.sha1(ref_src.encode()).hexdigest()
        with self._ref_lock:
            cached = self._ref_cache.get(key)
        gpu = self._gpus.get()
        t1 = time.monotonic()
        try:
            res = self._run(gpu, ref_src, code, language, seeds, cached, task_id=task_id)
            if self.workers_per_gpu > 1 and "OutOfMemoryError" in res.get("feedback", ""):
                res = self._run(gpu, ref_src, code, language, seeds, cached, exclusive=True, task_id=task_id)
        finally:
            self._gpus.put(gpu)
        if cached is None and res.get("t_ref_ms") is not None:
            with self._ref_lock:
                self._ref_cache.setdefault(key, {k: res[k] for k in REF_KEYS if res.get(k) is not None})
        return apply_roofline({**res, "gpu_id": gpu, "queue_wait_s": t1 - t0, "eval_s": time.monotonic() - t1})

    def _run(self, gpu: str, ref_src: str, code: str, language: str, seeds: List[int],
             ref_cache: Optional[Dict] = None, exclusive: bool = False, task_id: str = "") -> Dict:
        with tempfile.TemporaryDirectory(prefix="kernel_env_") as tmp:
            req, out, code_path = Path(tmp, "request.json"), Path(tmp, "result.json"), Path(tmp, "candidate.py")
            code_path.write_text(code)
            req.write_text(json.dumps({
                "ref_src": ref_src, "code_path": str(code_path), "language": language, "seeds": seeds, "task_id": task_id,
                "atol": self.atol, "rtol": self.rtol, "warmup_ms": self.warmup_ms, "rep_ms": self.rep_ms,
                "ref_cache": ref_cache, "exclusive": exclusive,
                "gpu_lock": f"{tempfile.gettempdir()}/kernel_env_gpu{gpu}.lock" if self.workers_per_gpu > 1 else None,
            }))
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu,
                "HOME": tmp,
                "TRITON_CACHE_DIR": f"{tmp}/triton",
                "TORCH_EXTENSIONS_DIR": f"{tmp}/torch_ext",
            }
            # own process group so a timeout also kills compiler children (nvcc, ptxas)
            proc = subprocess.Popen(
                [sys.executable, "-I", str(WORKER), str(req), str(out)],
                cwd=tmp, env=env, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            try:
                _, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                return {"stage": "timeout", "feedback": f"Evaluation exceeded {self.timeout_s:.0f}s (hang or very slow kernel)"}
            if not out.exists():
                return {"stage": "runtime", "feedback": f"Worker crashed (exit {proc.returncode}):\n{stderr[-2000:]}"}
            return json.loads(out.read_text())


class MockEvaluator(Evaluator):
    """No-GPU stand-in for Mac dev/tests: real static checks, scripted (or default) results."""

    DEFAULT = {"stage": "timed", "correct": True, "max_abs_err": 0.0, "t_ref_ms": 1.0, "t_new_ms": 1.0,
               "speedup": 1.0, "feedback": "[mock] Correct. speedup 1.00x"}

    def __init__(self, results: Optional[List[Dict]] = None):
        super().__init__(gpus=["mock"])
        self.results = list(results or [])
        self.calls: List[Dict] = []

    def _run(self, gpu, ref_src, code, language, seeds, ref_cache=None, exclusive=False, task_id=""):
        self.calls.append({"code": code, "language": language, "seeds": seeds, "ref_cache": ref_cache, "task_id": task_id})
        return self.results.pop(0) if self.results else dict(self.DEFAULT)


@lru_cache(maxsize=None)
def default_evaluator() -> Evaluator:
    """Process-wide evaluator. KERNEL_ENV_EVALUATOR=mock for machines without a CUDA GPU."""
    return MockEvaluator() if os.environ.get("KERNEL_ENV_EVALUATOR") == "mock" else Evaluator()

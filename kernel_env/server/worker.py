"""Evaluate one candidate kernel in an isolated process (GPU only).

Usage: python -I worker.py request.json result.json

The parent (evaluator.py) runs this with a timeout and a single visible GPU; crashes, hangs and
CUDA context corruption die with the process. Result is written to a file, not stdout, so
candidate prints cannot forge it. Candidate source has already passed static_checks.

GPU sharing: several workers may run on one GPU. Each holds a shared flock on the GPU's lock file
while it compiles and checks correctness, and an exclusive one while benchmarking, so no other
eval's kernels run on that GPU during a timing.
"""

import contextlib
import fcntl
import json
import sys
import time
import traceback

PHASES: dict = {}  # seconds per phase, returned with the result: where an eval's time went


def _tb(limit: int = 2000) -> str:
    return traceback.format_exc()[-limit:]


def _stage_for(e: BaseException) -> str:
    return "compile" if "CompilationError" in type(e).__name__ else "runtime"


class Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.monotonic()

    def __exit__(self, *exc):
        PHASES[self.name] = PHASES.get(self.name, 0.0) + time.monotonic() - self.t


def evaluate(req: dict) -> dict:
    with Phase("import"):
        import torch
        from triton.testing import do_bench  # bound before candidate import

    dev = torch.device("cuda")
    seeds = req["seeds"]
    lock = open(req["gpu_lock"], "a") if req.get("gpu_lock") else None

    @contextlib.contextmanager
    def exclusive():
        if lock is None:
            yield
            return
        # release before re-locking: an in-place shared->exclusive upgrade deadlocks when two workers
        # both hold the shared lock (seen under gVisor on Modal)
        with Phase("lock_wait"):
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits for other evals to leave the GPU
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, base)

    # exclusive=True: retry of a run that hit OOM while sharing the GPU; hold the GPU alone throughout
    base = fcntl.LOCK_EX if req.get("exclusive") else fcntl.LOCK_SH
    if lock is not None:
        with Phase("lock_wait"):
            fcntl.flock(lock, base)

    def to_dev(x):
        return x.to(dev) if isinstance(x, torch.Tensor) else x

    def bench(model, inputs):
        with Phase("bench"), exclusive(), torch.no_grad():
            return do_bench(lambda: model(*inputs), warmup=req["warmup_ms"], rep=req["rep_ms"], return_mode="median")

    ref_ns: dict = {}
    exec(compile(req["ref_src"], "reference.py", "exec"), ref_ns)
    with Phase("reference_init"), torch.device(dev):  # KernelBench builds tensors on CPU; build them on the GPU
        init_inputs = [to_dev(x) for x in ref_ns["get_init_inputs"]()]
        torch.manual_seed(seeds[0])
        ref = ref_ns["Model"](*init_inputs).to(dev).eval()

    def make_inputs(seed):
        with Phase("inputs"), torch.device(dev):
            torch.manual_seed(seed)
            return [to_dev(x) for x in ref_ns["get_inputs"]()]

    # Reference timing + roofline counts are cached per task by the evaluator (ref_cache) after the first run.
    # Otherwise time the reference before candidate code is imported, so import-time side effects can't slow it.
    out = dict(req["ref_cache"]) if req.get("ref_cache") else _measure_reference(torch, ref, make_inputs(seeds[0]), bench)
    out["gpu"] = torch.cuda.get_device_name()
    return _evaluate_candidate(req, torch, dev, ref, init_inputs, make_inputs, bench, out)


def _measure_reference(torch, ref, inputs, bench) -> dict:
    out = {"t_ref_ms": bench(ref, inputs)}
    # roofline inputs: counted FLOPs (matmul/conv/attention) and compulsory bytes (inputs + params + outputs)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            ref_out = ref(*inputs)
        outs = ref_out if isinstance(ref_out, (tuple, list)) else [ref_out]
        read = [x for x in inputs if isinstance(x, torch.Tensor)] + list(ref.parameters()) + list(ref.buffers())
        out["flops"] = fc.get_total_flops()
        out["bytes"] = sum(t.numel() * t.element_size() for t in read + [o for o in outs if isinstance(o, torch.Tensor)])
        out["dtype"] = next((str(x.dtype) for x in read if x.is_floating_point()), None)
    except Exception:
        pass  # roofline is optional; never fail an evaluation over it
    return out


def _evaluate_candidate(req, torch, dev, ref, init_inputs, make_inputs, bench, out) -> dict:
    atol, rtol, seeds = req["atol"], req["rtol"], req["seeds"]
    cand_ns: dict = {"__name__": "candidate"}
    try:
        # compile from a real file: @triton.jit reads kernel source via inspect.getsource
        with Phase("import_candidate"), open(req["code_path"]) as f:
            exec(compile(f.read(), req["code_path"], "exec"), cand_ns)
    except BaseException:
        return {**out, "stage": "compile", "feedback": _tb()}
    try:
        torch.manual_seed(seeds[0])  # same seed => same parameter init as reference
        with torch.device(dev):
            new = cand_ns["ModelNew"](*init_inputs).to(dev).eval()
    except BaseException as e:
        return {**out, "stage": _stage_for(e), "feedback": _tb()}

    max_err = 0.0
    call_s = []  # wall time of each candidate call in the correctness trials (trial 0 includes JIT)
    for k, seed in enumerate(seeds):
        inputs = make_inputs(seed)
        # the first call includes Triton JIT / CUDA compile time
        with Phase("correctness" if k else "first_call_incl_jit"), torch.no_grad():
            ref_out = ref(*inputs)
            sums = [torch.sum(x, dtype=torch.float64).item() for x in inputs if isinstance(x, torch.Tensor)]
            try:
                t0 = time.monotonic()
                new_out = new(*inputs)
                torch.cuda.synchronize()
                call_s.append(time.monotonic() - t0)
            except BaseException as e:
                return {**out, "stage": _stage_for(e), "feedback": _tb()}

        refs = list(ref_out) if isinstance(ref_out, (tuple, list)) else [ref_out]
        news = list(new_out) if isinstance(new_out, (tuple, list)) else [new_out]
        if len(refs) != len(news):
            return {**out, "stage": "correctness", "correct": False,
                    "feedback": f"Expected {len(refs)} outputs, got {len(news)}"}
        in_ptrs = {x.untyped_storage().data_ptr() for x in inputs if isinstance(x, torch.Tensor)}
        for i, (r, n) in enumerate(zip(refs, news)):
            if not isinstance(n, torch.Tensor) or n.shape != r.shape or n.dtype != r.dtype or n.device != r.device:
                got = (tuple(n.shape), n.dtype, n.device) if isinstance(n, torch.Tensor) else type(n).__name__
                return {**out, "stage": "correctness", "correct": False,
                        "feedback": f"Output {i}: expected {(tuple(r.shape), r.dtype, r.device)}, got {got}"}
            if n.untyped_storage().data_ptr() in in_ptrs:
                return {**out, "stage": "correctness", "correct": False,
                        "feedback": f"Output {i} aliases an input tensor; allocate a new output"}
            err = (n.float() - r.float()).abs().nan_to_num(float("inf")).max().item() if n.numel() else 0.0
            max_err = max(max_err, err)
            if not torch.allclose(n, r, atol=atol, rtol=rtol, equal_nan=True):
                bad = (~torch.isclose(n, r, atol=atol, rtol=rtol, equal_nan=True)).sum().item()
                return {**out, "stage": "correctness", "correct": False, "max_abs_err": err,
                        "feedback": f"Output {i} mismatch on trial {k}: {bad}/{n.numel()} elements wrong, "
                                    f"max_abs_err={err:.3g} (atol={atol}, rtol={rtol})"}
        if sums != [torch.sum(x, dtype=torch.float64).item() for x in inputs if isinstance(x, torch.Tensor)]:
            return {**out, "stage": "correctness", "correct": False,
                    "feedback": "Inputs were modified in place; kernels must not write to inputs"}
        del inputs, ref_out, new_out, refs, news

    t_ref = out["t_ref_ms"]
    # A kernel far slower than the reference would hold the GPU's exclusive lock for minutes in do_bench while
    # every other eval waits; its reward is ~0.1 regardless, so use the trial timing instead.
    trial_ms = 1000 * min(call_s[1:] or call_s)
    if trial_ms > max(req.get("slow_factor", 50) * t_ref, 1000.0):
        t_new, note = trial_ms, " (very slow: timed from the correctness trials, not benchmarked)"
    else:
        inputs = make_inputs(seeds[0])
        t_new, note = bench(new, inputs), ""
    speedup = t_ref / t_new
    return {**out, "stage": "timed", "correct": True, "max_abs_err": max_err, "t_new_ms": t_new,
            "speedup": speedup,
            "feedback": f"Correct on {len(seeds)} trials. Reference {t_ref:.4f} ms, yours {t_new:.4f} ms{note}, "
                        f"speedup {speedup:.2f}x on {out['gpu']}."}


def main(req_path: str, out_path: str) -> None:
    with open(req_path) as f:
        req = json.load(f)
    try:
        res = evaluate(req)
    except BaseException:
        res = {"stage": "runtime", "feedback": _tb()}
    res["phases"] = {k: round(v, 3) for k, v in PHASES.items()}
    with open(out_path, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

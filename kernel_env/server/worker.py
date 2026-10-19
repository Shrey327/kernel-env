"""Evaluate one candidate kernel in an isolated process (GPU only).

Usage: python -I worker.py request.json result.json

The parent (evaluator.py) runs this with a timeout and a single visible GPU; crashes, hangs and
CUDA context corruption die with the process. Result is written to a file, not stdout, so
candidate prints cannot forge it. Candidate source has already passed static_checks.

GPU sharing: several workers may run on one GPU. Each holds a shared flock on the GPU's lock file
while it compiles and checks correctness, and an exclusive one while benchmarking, so no other
eval's kernels run on that GPU during a timing.
"""

import ast
import contextlib
import fcntl
import statistics
import json
import math
import re
import sys
import time
import traceback

try:
    from .validation_memory import input_fingerprint, paired_tensor_chunks, tensor_chunks, max_abs_error, CHUNK_BYTES
    from .benchmark_protocol import paired_timings, cuda_coverage
    from .cuda_graph_timing import capture_work, measure_work, input_batches, checked_measure, checked_eager_warmup, prepare_check, finish_check, check_outputs, calibrated_floor, tensor_bytes, graph_repeats, graph_memory_plan, cuda_memory_evidence, MEASUREMENT_SCOPE, METHOD as GRAPH_METHOD
except ImportError:
    import runpy
    from pathlib import Path
    _memory_helpers = runpy.run_path(str(Path(__file__).with_name("validation_memory.py")))
    input_fingerprint, paired_tensor_chunks, tensor_chunks, max_abs_error = (_memory_helpers[name] for name in ('input_fingerprint','paired_tensor_chunks','tensor_chunks','max_abs_error'))
    CHUNK_BYTES=_memory_helpers['CHUNK_BYTES']
    _benchmark_helpers = runpy.run_path(str(Path(__file__).with_name("benchmark_protocol.py")))
    paired_timings, cuda_coverage = _benchmark_helpers["paired_timings"], _benchmark_helpers["cuda_coverage"]
    _graph_helpers = runpy.run_path(str(Path(__file__).with_name("cuda_graph_timing.py")))
    capture_work, measure_work, input_batches, checked_measure, checked_eager_warmup, prepare_check, finish_check, check_outputs, calibrated_floor = (_graph_helpers[name] for name in ('capture_work','measure_work','input_batches','checked_measure','checked_eager_warmup','prepare_check','finish_check','check_outputs','calibrated_floor'))
    GRAPH_METHOD = _graph_helpers['METHOD']
    tensor_bytes, graph_repeats, graph_memory_plan, cuda_memory_evidence, MEASUREMENT_SCOPE = (_graph_helpers[name] for name in ('tensor_bytes','graph_repeats','graph_memory_plan','cuda_memory_evidence','MEASUREMENT_SCOPE'))

PHASES: dict = {}  # seconds per phase, returned with the result: where an eval's time went

SOFTMAX = "level1/23_Softmax"
FROBENIUS = "level1/37_FrobeniusNorm_"
NORMALIZED_TASKS = {SOFTMAX, FROBENIUS}
ACTIVATION_TASKS = {'level1/19_ReLU', 'level1/20_LeakyReLU', 'level1/31_ELU',
                    'level1/29_Softplus', 'level1/88_MinGPTNewGelu', 'level1/21_Sigmoid',
                    'level1/22_Tanh', 'level1/25_Swish', 'level1/26_GELU_', 'level1/27_SELU_',
                    'level1/28_HardSigmoid', 'level1/30_Softsign', 'level1/32_HardTanh'}
REDUCTION_TASKS = {'level1/47_Sum_reduction_over_a_dimension', 'level1/48_Mean_reduction_over_a_dimension'}
REGRESSION_LOSSES = {'level1/94_MSELoss', 'level1/96_HuberLoss'}


class HostOperationError(RuntimeError):
    pass


def kernel_only_mode(device_type="cuda"):
    """Catch tensor operators that AST checks cannot distinguish from scalar/shape arithmetic.

    This checks PyTorch dispatch inside forward, not arbitrary native code or graphs built in __init__.
    Allocation, layout and device-local copies stay allowed. CPU calibration math in Triton's
    autotuner stays outside the guarded device; candidate inputs cannot be copied there for compute.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    import torch

    allowed = {
        "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided",
        "zeros", "zeros_like", "new_zeros", "ones", "ones_like", "full", "full_like", "new_full",
        "arange", "rand", "randn", "randint", "rand_like", "randn_like",
        "view", "_unsafe_view", "reshape", "as_strided", "slice", "select", "expand",
        "transpose", "t", "permute", "squeeze", "unsqueeze", "detach", "alias",
        "clone", "contiguous", "to", "_to_copy", "copy_", "zero_", "fill_", "lift_fresh",
    }

    class KernelOnlyMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            namespace, name = func._schema.name.split("::", 1)
            touches_device = any(isinstance(x, torch.Tensor) and x.device.type == device_type
                                 for x in tree_leaves((args, kwargs)))
            if namespace == "aten" and touches_device:
                off_device = (name == "_to_copy" and kwargs.get("device") is not None
                              and torch.device(kwargs["device"]).type != device_type)
                off_device |= name == "copy_" and args[0].device.type != device_type
                if name not in allowed or off_device:
                    raise HostOperationError(f"PyTorch operation '{func}' in forward; use a custom kernel")
            return func(*args, **kwargs)

    return KernelOnlyMode()


def adversarial_inputs(torch, inputs, task_id: str, trial: int):
    """Keep task shapes; vary normalization inputs as well as RNG seeds. Also used as fresh replay challenges outside timing."""
    if task_id in REGRESSION_LOSSES:
        if trial == 1:
            inputs[1].copy_(inputs[0])
        elif trial == 2:
            inputs[0].zero_()
            inputs[1].fill_(2)
        return inputs
    if task_id in REDUCTION_TASKS:
        if trial == 1:
            inputs[0].zero_()
        elif trial == 2:
            inputs[0].mul_(8).sub_(2)
        return inputs
    if task_id in ACTIVATION_TASKS:
        if trial == 1:
            inputs[0].mul_(64).sub_(32)
        elif trial == 2:
            inputs[0].zero_()
        return inputs
    if task_id not in NORMALIZED_TASKS:
        return inputs
    x = inputs[0]
    if trial == 1:
        x.fill_(0 if task_id == SOFTMAX else 1)
    elif trial == 2:
        if task_id == SOFTMAX:
            x.zero_()
            rows = torch.arange(x.shape[0], device=x.device)
            x[rows, (rows * 104729) % x.shape[1]] = math.log(x.shape[1]) + 8
        else:
            x.mul_(2).sub_(1)
            scales = torch.arange(1, x.shape[0] + 1, device=x.device)
            x.mul_(scales.reshape((-1,) + (1,) * (x.ndim - 1)))
    elif trial == 3 and task_id == FROBENIUS:
        x.zero_()
        x[(0,) * x.ndim] = 3
        x[(-1,) * x.ndim] = -4
    return inputs


def numerical_error(torch, reference, actual, task_id: str, atol: float, rtol: float):
    """Return feedback on failure. Other tasks retain their configured tolerance (including TF32)."""
    if task_id in NORMALIZED_TASKS:
        largest=0.0
        for ref,out in paired_tensor_chunks(reference,actual):
            if not torch.isfinite(ref).all() or not torch.isfinite(out).all():
                return "Normalization output must be finite"
            if ref.numel():largest=max(largest,ref.abs().max().item())
        # Absolute 0.01 dwarfs these outputs. Scale its floor to the reference instead,
        # retaining the configured relative tolerance and a small allowance near zero.
        atol = min(atol, largest * 1e-5)
        if task_id == SOFTMAX:
            # Keep row sums, float64 accumulation and original tolerance, with bounded slices.
            elements=max(1,CHUNK_BYTES//actual.element_size())
            rows_per_chunk=max(1,elements//max(1,actual.shape[1]))
            for row in range(0,actual.shape[0],rows_per_chunk):
                rows=actual[row:row+rows_per_chunk]
                total=torch.zeros(rows.shape[0],device=actual.device,dtype=torch.float64)
                columns_per_chunk=max(1,elements//max(1,rows.shape[0]))
                for column in range(0,actual.shape[1],columns_per_chunk):
                    chunk=rows[:,column:column+columns_per_chunk]
                    if (chunk<0).any():
                        return "Softmax probabilities must be nonnegative and each row must sum to 1"
                    total.add_(chunk.sum(dim=1,dtype=torch.float64))
                if not torch.allclose(total,torch.ones_like(total),atol=1e-4,rtol=rtol):
                    return "Softmax probabilities must be nonnegative and each row must sum to 1"
        else:
            norm_squared=torch.zeros((),device=actual.device,dtype=torch.float64)
            for chunk in tensor_chunks(actual):norm_squared.add_(chunk.to(torch.float64).square().sum())
            if not math.isclose(math.sqrt(norm_squared.item()),1.0,abs_tol=1e-4,rel_tol=rtol):
                return "Frobenius normalization must have global L2 norm 1 (not one norm per slice)"
    bad=0
    for ref,out in paired_tensor_chunks(reference,actual):
        bad+=(~torch.isclose(out,ref,atol=atol,rtol=rtol,equal_nan=task_id not in NORMALIZED_TASKS)).sum().item()
    if bad:
        return f"{bad}/{actual.numel()} elements wrong (atol={atol:.3g}, rtol={rtol})"
    return None


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
    torch.cuda.reset_peak_memory_stats(dev)
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

    synchronize = torch.cuda.synchronize
    stream = torch.cuda.current_stream(dev)
    stream_context = torch.cuda.stream

    def measure(model, inputs):
        work=capture_work(torch,model,inputs,dev)
        return measure_work(torch,work,dev)

    def bench(model, inputs, samples=None):
        with Phase("bench"), exclusive(), torch.no_grad():
            trusted=ref(*inputs)
            values=list(trusted) if isinstance(trusted,(tuple,list)) else [trusted]
            size=tensor_bytes(torch,inputs)
            graph_memory_plan(size,tensor_bytes(torch,values),graph_repeats(size))
            del trusted,values
            work=capture_work(torch,model,inputs,dev)
            cuda_memory_evidence(torch,dev)
            medians=[measure_work(torch,work,dev) for _ in range(3)]
            if samples is not None:samples.extend(medians)
            return statistics.median(medians)

    def paired(ref, candidate, inputs, names):
        with Phase('paired_bench'),exclusive(),torch.no_grad():
            trusted=ref(*inputs)
            trusted_values=list(trusted) if isinstance(trusted,(tuple,list)) else [trusted]
            size=tensor_bytes(torch,inputs)
            memory_plan=graph_memory_plan(size,tensor_bytes(torch,trusted_values),graph_repeats(size))
            del trusted,trusted_values
            batches=input_batches(torch,inputs)
            fingerprint=lambda xs:input_fingerprint(torch,xs)
            error=lambda a,b:numerical_error(torch,a,b,req.get('task_id',''),req['atol'],req['rtol'])
            warmup=lambda model,xs:checked_eager_warmup(torch,model,xs,ref,fingerprint,error)
            reference=capture_work(torch,ref,inputs,dev,batches=batches,warmup_check=warmup)
            # Captured kernels do not execute during capture, so anything profiled here ran eagerly: warmups on
            # the capture stream, or capture-time work that escaped the graph on a side stream (rejected).
            synchronize(dev)
            with profile_class(activities=[cuda_activity]) as capture_trace:
                generated=capture_work(torch,candidate,inputs,dev,batches=batches,warmup_check=warmup)
                synchronize(dev)
            capture_profile=cuda_coverage(capture_trace.events(),names)
            cuda_memory_evidence(torch,dev)
            count=[0]
            def validated(work):
                # Each paired round sees identical values; subsequent rounds get fresh values.
                round_id=count[0]//2;count[0]+=1
                fresh=[adversarial_inputs(torch,make_inputs(seeds[0]+round_id*1000003+i*65537),
                         req.get('task_id',''),(round_id+i)%len(seeds)) for i in range(work.repeats)]
                measured=checked_measure(torch,work,fresh,ref,fingerprint,error,dev)
                cuda_memory_evidence(torch,dev)
                return measured
            # Actual graph replay must contain generated CUDA work; all checks remain outside events.
            fresh=[adversarial_inputs(torch,make_inputs(seeds[0]+i*65537),req.get('task_id',''),i%len(seeds)) for i in range(generated.repeats)]
            expected,before=prepare_check(torch,generated,fresh,ref,fingerprint)
            replay_profile=profile(generated.replay,[],names)
            finish_check(torch,generated,expected,before,fingerprint,error)
            del fresh,expected
            if replay_profile['generated_cuda_fraction']<=0:
                raise HostOperationError('No generated CUDA work in captured replay')
            floor,calibration=calibrated_floor(torch,generated,dev)
            measurement=paired_timings(lambda:validated(reference),lambda:validated(generated),
                                      seeds[0],absolute_floor_ms=floor)
            measurement.update(timing_method=GRAPH_METHOD,
                measurement_scope=MEASUREMENT_SCOPE,memory_plan=memory_plan,
                cuda_memory=cuda_memory_evidence(torch,dev),
                graph_repeats=generated.repeats,validated_graph_warmups=3,timing_calibration=calibration,
                warmup_evidence={'reference_eager':reference.eager_warmup_evidence,'candidate_eager':generated.eager_warmup_evidence,
                                 'reference_graph':reference.warmup_evidence,'candidate_graph':generated.warmup_evidence},replay_profiling_summary=json.dumps(replay_profile),
                capture_profiling_summary=json.dumps(capture_profile))
            return measurement

    profile_class = torch.profiler.profile
    cuda_activity = torch.profiler.ProfilerActivity.CUDA

    def profile(candidate, inputs, names):
        with Phase("profile"), exclusive(), torch.no_grad():
            if inputs:
                expected=ref(*inputs);before=input_fingerprint(torch,inputs)
            synchronize(dev)
            with profile_class(activities=[cuda_activity]) as trace:
                output=candidate(*inputs)
                synchronize(dev)
            if inputs:
                if before!=input_fingerprint(torch,inputs):raise HostOperationError('profiled execution modified inputs')
                check_outputs(torch,expected,output,inputs,lambda a,b:numerical_error(torch,a,b,req.get('task_id',''),req['atol'],req['rtol']))
        return cuda_coverage(trace.events(), names)

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
    out = _measure_reference(torch, ref, make_inputs(seeds[0]), bench)
    out["gpu"] = torch.cuda.get_device_name()
    result=_evaluate_candidate(req, torch, dev, ref, init_inputs, make_inputs, bench, out, paired=paired, profile=profile)
    result['cuda_memory']=cuda_memory_evidence(torch,dev)
    return result


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


def _evaluate_candidate(req, torch, dev, ref, init_inputs, make_inputs, bench, out, paired=None, profile=None) -> dict:
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

    guard = kernel_only_mode(dev.type)

    def candidate(*inputs):
        with guard:
            return new(*inputs)

    max_err = 0.0
    call_s = []  # wall time of each candidate call in the correctness trials (trial 0 includes JIT)
    for k, seed in enumerate(seeds):
        inputs = adversarial_inputs(torch, make_inputs(seed), req.get("task_id", ""), k)
        # the first call includes Triton JIT / CUDA compile time
        with Phase("correctness" if k else "first_call_incl_jit"), torch.no_grad():
            ref_out = ref(*inputs)
            with Phase("input_mutation_checks"):
                fingerprints = input_fingerprint(torch, inputs)
            try:
                t0 = time.monotonic()
                new_out = candidate(*inputs)
                torch.cuda.synchronize()
                if dev.type=='cuda':cuda_memory_evidence(torch,dev)
                call_s.append(time.monotonic() - t0)
            except HostOperationError as e:
                return {**out, "stage": "format", "correct": False, "feedback": str(e)}
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
            err = max_abs_error(torch,r,n)
            max_err = max(max_err, err)
            error = numerical_error(torch, r, n, req.get("task_id", ""), atol, rtol)
            if error:
                return {**out, "stage": "correctness", "correct": False, "max_abs_err": err,
                        "feedback": f"Output {i} mismatch on trial {k}: {error}; max_abs_err={err:.3g}"}
        with Phase("input_mutation_checks"):
            mutated = fingerprints != input_fingerprint(torch, inputs)
        if mutated:
            return {**out, "stage": "correctness", "correct": False,
                    "feedback": "Inputs were modified in place; kernels must not write to inputs"}
        del inputs, ref_out, new_out, refs, news, r, n

    t_ref = out["t_ref_ms"]
    # A kernel far slower than the reference would hold the GPU's exclusive lock for minutes in do_bench while
    # every other eval waits; its reward is ~0.1 regardless, so use the trial timing instead.
    trial_ms = 1000 * min(call_s[1:] or call_s)
    timing_samples = []
    if paired is not None and trial_ms > max(req.get("slow_factor",50)*t_ref,1000.0):
        return {**out,'stage':'timeout','correct':False,
                'feedback':'Correctness passed but eager runtime exceeds graph admission bound; no alternate timer or performance claim'}
    if trial_ms > max(req.get("slow_factor", 50) * t_ref, 1000.0):
        t_new, note = trial_ms, " (very slow: timed from the correctness trials, not benchmarked)"
        timing_samples = [1000 * value for value in call_s[1:] or call_s]
        timing_method = "correctness_wall_slow"
        out.update(speedup_lower_bound=0.0, speedup_upper_bound=t_ref/max(t_new,1e-9))
    else:
        inputs = make_inputs(seeds[0])
        try:
            if profile is not None:
                source = open(req['code_path']).read()
                names = {node.name for node in ast.walk(ast.parse(source))
                         if isinstance(node, ast.FunctionDef) and node.decorator_list}
                names.update(re.findall(r'__global__\s+void\s+(\w+)',source))
                summaries=[]
                for training in (False, True):
                    new.train(training)
                    summaries.append(profile(candidate,inputs,names))
                new.eval()
                if any(item['generated_cuda_fraction'] <= 0 for item in summaries):
                    return {**out,'stage':'format','correct':False,
                            'feedback':'No executed generated CUDA kernel in eval or train mode'}
                out['generated_cuda_fraction'] = summaries[0]['generated_cuda_fraction']
                out['profiling_summary'] = json.dumps(summaries[0])
            if paired is not None:
                measurement = paired(ref, candidate, inputs, names)
                out.update(measurement)
                t_ref = measurement['t_ref_ms']
                t_new = measurement['t_new_ms']
                timing_samples = measurement['timing_samples_ms']
                timing_method = measurement['timing_method']
                note = " (amortized CUDA graph replay event interval; launch overhead included; host checks outside timing)"
            else:
                t_new, note = bench(candidate, inputs, timing_samples), ""
                timing_method = "three_warmed_benchmark_medians"
        except (HostOperationError,ValueError) as e:
            return {**out, "stage": "format", "correct": False, "feedback": str(e)}
    speedup = t_ref / t_new
    return {**out, "stage": "timed", "correct": True, "max_abs_err": max_err, "t_new_ms": t_new,
            "timing_samples_ms": timing_samples, "timing_method": timing_method,
            "speedup": speedup,
            "feedback": f"Correct on {len(seeds)} trials. Reference {t_ref:.4f} ms, yours {t_new:.4f} ms{note}, "
                        f"speedup {speedup:.2f}x on {out['gpu']}."}


def main(req_path: str, out_path: str) -> None:
    with open(req_path) as f:
        req = json.load(f)
    try:
        res = evaluate(req)
    except BaseException:
        res = {"stage": "infrastructure", "correct": False, "feedback": "Evaluator infrastructure failure:\n" + _tb()}
    res["phases"] = {k: round(v, 3) for k, v in PHASES.items()}
    with open(out_path, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

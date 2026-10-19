"""Amortized CUDA graph replay event interval, with unsubtracted launch overhead."""
import statistics,time,hashlib
from dataclasses import dataclass,field

METHOD='fresh_paired_cuda_graph_event_interval_v5'
MEASUREMENT_SCOPE='amortized CUDA graph replay event interval with unsubtracted launch overhead; host validation/guards outside events; captured GPU copies included'
GRAPH_INPUT_TARGET_BYTES=512*1024**2
LIVE_MEMORY_CAP_BYTES=64*1024**3
MEMORY_RESERVE_BYTES=8*1024**3
COMPARISON_SCRATCH_BYTES=256*1024**2
EAGER_WARMUP_CALLS=1
VALIDATED_GRAPH_WARMUPS=3


class GraphCaptureError(ValueError):
    pass


@dataclass
class CapturedWork:
    graph: object
    outputs: list
    batches: list
    stream: object
    warmed: bool = False
    warmup_evidence: list = field(default_factory=list)
    eager_warmup_evidence: list = field(default_factory=list)
    torch_api: object = None

    @property
    def repeats(self):return len(self.batches)

    def replay(self):
        if self.torch_api is None:self.graph.replay()
        else:
            with self.torch_api.cuda.stream(self.stream):self.graph.replay()


def graph_repeats(input_bytes):
    return min(64,max(1,GRAPH_INPUT_TARGET_BYTES//max(1,input_bytes)))


def tensor_bytes(torch,values):
    return sum(x.numel()*x.element_size() for x in values if isinstance(x,torch.Tensor))


def graph_memory_plan(input_bytes,output_bytes,repeats):
    # Includes original inputs, shared captured/fresh inputs, both retained graph outputs,
    # fresh reference outputs, two input-sized reference workspaces, and chunk scratch.
    live=input_bytes+repeats*(4*input_bytes+3*output_bytes)+COMPARISON_SCRATCH_BYTES
    plan={'input_bytes':input_bytes,'trusted_output_bytes':output_bytes,'repeats':repeats,
          'estimated_live_bytes':live,'reserve_bytes':MEMORY_RESERVE_BYTES,
          'live_cap_bytes':LIVE_MEMORY_CAP_BYTES,'scratch_bytes':COMPARISON_SCRATCH_BYTES,
          'limitation':'reference workspace allowance is two input sizes per invocation; arbitrary candidate workspace is not bounded by this estimate; actual CUDA peaks are gated'}
    if live+MEMORY_RESERVE_BYTES>LIVE_MEMORY_CAP_BYTES:
        raise ValueError('graph memory estimate exceeds live cap including reserve: '+str(plan))
    return plan


def cuda_memory_evidence(torch,device):
    allocated=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
    evidence={'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
              'live_cap_bytes':LIVE_MEMORY_CAP_BYTES,'reserve_bytes':MEMORY_RESERVE_BYTES}
    if max(allocated,reserved)+MEMORY_RESERVE_BYTES>LIVE_MEMORY_CAP_BYTES:
        raise ValueError('actual CUDA memory peak exceeds live cap including reserve: '+str(evidence))
    return evidence


def input_batches(torch,inputs):
    repeats=graph_repeats(tensor_bytes(torch,inputs))
    return [[x.clone() if isinstance(x,torch.Tensor) else x for x in inputs] for _ in range(repeats)]


def capture_work(torch,model,inputs,device,batches=None,warmup_check=None):
    batches=input_batches(torch,inputs) if batches is None else batches
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    eager_evidence=[]
    with torch.cuda.stream(stream),torch.no_grad():
        for index in range(EAGER_WARMUP_CALLS):
            began=time.monotonic()
            if warmup_check:warmup_check(model,inputs)
            else:model(*inputs)
            eager_evidence.append({'index':index+1,'validated':bool(warmup_check),'wall_seconds_including_checks':time.monotonic()-began})
    torch.cuda.synchronize(device)
    graph=torch.cuda.CUDAGraph();outputs=[]
    try:
        with torch.cuda.graph(graph,stream=stream,capture_error_mode='global'),torch.no_grad():
            for batch in batches:outputs.append(model(*batch))
    except RuntimeError as error:
        raise GraphCaptureError('GPU graph capture unsupported or invalid; no alternate timer: '+str(error)[:300]) from error
    torch.cuda.synchronize(device)
    # Holding every output prevents early-wrong/last-correct capture branches from hiding.
    seen=set()
    for output in outputs:
        values=output if isinstance(output,(tuple,list)) else [output]
        for value in values:
            if isinstance(value,torch.Tensor):
                pointer=value.untyped_storage().data_ptr()
                if pointer in seen:raise GraphCaptureError('Captured invocations reuse output storage; independent fresh outputs required')
                seen.add(pointer)
    work=CapturedWork(graph,outputs,batches,stream,torch_api=torch);work.eager_warmup_evidence=eager_evidence
    return work


def measure_work(torch,work,device):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(work.stream):
        start.record()
        work.graph.replay()
        end.record()
    end.synchronize()
    return start.elapsed_time(end)/work.repeats


def calibrated_floor(torch,work,device):
    # Null graph uses the same event/launch path. Never subtract overhead from reported times.
    null=torch.cuda.CUDAGraph()
    with torch.cuda.graph(null,stream=work.stream,capture_error_mode='global'):pass
    empty=CapturedWork(null,[None]*work.repeats,work.batches,work.stream,torch_api=torch)
    canary=torch.arange(16,device=device);saved=canary.clone()
    null_warmups=[]
    for index in range(VALIDATED_GRAPH_WARMUPS):
        began=time.monotonic();empty.replay();torch.cuda.synchronize(device)
        if not torch.equal(canary,saved):raise ValueError('null graph modified independent canary')
        null_warmups.append({'index':index+1,'canary_unchanged':True,'wall_seconds_including_check':time.monotonic()-began})
    samples=[measure_work(torch,empty,device) for _ in range(7)]
    if not torch.equal(canary,saved):raise ValueError('null calibration modified independent canary')
    # CUDA event resolution is approximately0.5us per complete graph, amortized per invocation.
    event_floor=.0005/work.repeats;null_median=statistics.median(samples);null_range=max(samples)-min(samples)
    floor=max(event_floor,null_median,null_range)
    return floor,{'null_graph_per_invocation_ms':samples,'absolute_floor_ms':floor,'repeats':work.repeats,
                  'warmup_evidence':null_warmups,'warmup_replays':VALIDATED_GRAPH_WARMUPS,'null_validation':'trusted empty graph with no model inputs/outputs; independent canary checked',
                  'event_resolution_per_invocation_ms':event_floor,'null_median_per_invocation_ms':null_median,
                  'null_range_per_invocation_ms':null_range,'measurement_scope':MEASUREMENT_SCOPE,
                  'kind':'engineering bound from unsubtracted null event interval (including submission/idle gaps), full range and event resolution; not confidence interval'}


def check_outputs(torch,expected,actual,inputs,error_check):
    refs=list(expected) if isinstance(expected,(list,tuple)) else [expected]
    outs=list(actual) if isinstance(actual,(list,tuple)) else [actual]
    if len(refs)!=len(outs):raise ValueError('captured output count mismatch')
    pointers={x.untyped_storage().data_ptr() for x in inputs if isinstance(x,torch.Tensor)}
    for ref,out in zip(refs,outs):
        if (not isinstance(out,torch.Tensor) or out.shape!=ref.shape or out.dtype!=ref.dtype or out.device!=ref.device):
            raise ValueError('captured output metadata mismatch')
        if out.untyped_storage().data_ptr() in pointers:raise ValueError('captured output aliases input')
        error=error_check(ref,out)
        if error:raise ValueError('captured output mismatch: '+error)


def prepare_check(torch,work,fresh_batches,reference,fingerprint):
    if len(work.batches)!=len(fresh_batches):raise ValueError('captured batch count mismatch')
    expected=[]
    for old,fresh in zip(work.batches,fresh_batches):
        if len(old)!=len(fresh):raise ValueError('captured input count mismatch')
        for target,value in zip(old,fresh):
            if isinstance(target,torch.Tensor):
                if not isinstance(value,torch.Tensor) or target.shape!=value.shape or target.dtype!=value.dtype:
                    raise ValueError('captured input signature changed')
                target.copy_(value)
            elif target!=value:raise ValueError('captured non-tensor input changed')
        expected.append(reference(*fresh))
    flat=[x for batch in work.batches for x in batch]
    before=fingerprint(flat)
    # Copies/reference work may run on the default stream; replay uses the capture stream.
    work.stream.wait_stream(torch.cuda.current_stream())
    return expected,before


def finish_check(torch,work,expected,before,fingerprint,error_check):
    flat=[x for batch in work.batches for x in batch]
    if before!=fingerprint(flat):raise ValueError('captured graph modified inputs')
    for batch,want,actual in zip(work.batches,expected,work.outputs):
        check_outputs(torch,want,actual,batch,error_check)


def checked_measure(torch,work,fresh_batches,reference,fingerprint,error_check,device):
    expected,before=prepare_check(torch,work,fresh_batches,reference,fingerprint)
    if not work.warmed:
        for index in range(VALIDATED_GRAPH_WARMUPS):
            began=time.monotonic();work.replay();torch.cuda.synchronize(device)
            finish_check(torch,work,expected,before,fingerprint,error_check)
            work.warmup_evidence.append({'index':index+1,'outputs_and_inputs_validated':True,
                 'input_fingerprint_sha256':hashlib.sha256(repr(before).encode()).hexdigest(),
                 'wall_seconds_including_checks':time.monotonic()-began})
        work.warmed=True
    value=measure_work(torch,work,device)
    # Every measured replay is checked; there is no unvalidated timing-only call path.
    finish_check(torch,work,expected,before,fingerprint,error_check)
    return value


def checked_eager_warmup(torch,model,inputs,reference,fingerprint,error_check):
    expected=reference(*inputs);before=fingerprint(inputs)
    output=model(*inputs);torch.cuda.synchronize()
    if before!=fingerprint(inputs):raise ValueError('eager warmup modified inputs')
    check_outputs(torch,expected,output,inputs,error_check)
    return output

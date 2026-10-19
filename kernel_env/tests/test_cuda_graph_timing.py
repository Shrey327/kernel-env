import contextlib
from types import SimpleNamespace
import pytest
import torch
from kernel_env.server.cuda_graph_timing import capture_work,measure_work,checked_measure,CapturedWork,GraphCaptureError


def fake_torch():
    events=[]
    class Event:
        def __init__(self,**kwargs):self.number=sum(x=='new_event' for x in events);events.append('new_event')
        def record(self):events.append('start' if self.number%2==0 else 'end')
        def synchronize(self):events.append('event_sync')
        def elapsed_time(self,other):return 256.
    class Graph:
        def replay(self):events.append('gpu_graph_replay')
    @contextlib.contextmanager
    def stream(*args,**kwargs):
        events.append('stream_enter');yield;events.append('stream_exit')
    @contextlib.contextmanager
    def graph(*args,**kwargs):
        assert kwargs['capture_error_mode']=='global'
        events.append('capture_enter');yield;events.append('capture_exit')
    cuda=SimpleNamespace(Stream=lambda **kw:SimpleNamespace(wait_stream=lambda s:events.append('stream_wait')),current_stream=lambda dev=None:object(),
         stream=stream,graph=graph,CUDAGraph=Graph,Event=Event,synchronize=lambda *a:events.append('device_sync'))
    return SimpleNamespace(Tensor=torch.Tensor,no_grad=torch.no_grad,cuda=cuda,arange=lambda n,**kw:torch.arange(n),equal=torch.equal),events


def test_host_guard_and_model_never_run_inside_measured_events():
    api,events=fake_torch();inputs=[torch.ones(4)]
    def guarded_model(x):events.extend(['guard_enter','python_model','guard_exit']);return x*2
    work=capture_work(api,guarded_model,inputs,'cuda')
    before=events.count('python_model')
    assert measure_work(api,work,'cuda')==4.
    assert events.count('python_model')==before
    measured=events[events.index('start')+1:events.index('end')]
    assert measured==['gpu_graph_replay']
    assert events.index('device_sync')<events.index('start')


def test_reference_and_candidate_use_identical_replay_path():
    api,events=fake_torch();inputs=[torch.ones(4)]
    ref=capture_work(api,lambda x:x+1,inputs,'cuda')
    candidate=capture_work(api,lambda x:x*2,inputs,'cuda')
    assert ref.repeats==candidate.repeats==64
    results=[]
    for work in (ref,candidate):
        events.clear();measure_work(api,work,'cuda');results.append(events[:])
    assert results[0]==results[1]


def work_for(kind):
    batches=[[torch.ones(3)],[torch.full((3,),2.)]]
    outputs=[x[0]*2 for x in batches];count=[0]
    def replay():
        count[0]+=1
        for batch,output in zip(batches,outputs):
            if kind=='fresh':output.copy_(batch[0]*2)
            elif kind=='late':output.copy_(batch[0]*2 if count[0]<=4 else torch.zeros(3))
            elif kind=='mutating':batch[0].zero_();output.zero_()
    return CapturedWork(SimpleNamespace(replay=replay),outputs,batches,SimpleNamespace(wait_stream=lambda s:None))


def test_every_measured_replay_checks_stale_late_switch_and_mutation():
    api,_=fake_torch();fresh=[[torch.full((3,),3.)],[torch.full((3,),7.)]]
    fingerprint=lambda xs:tuple(tuple(x.tolist()) for x in xs)
    error=lambda a,b:None if torch.equal(a,b) else 'values differ'
    ref=lambda x:x*2
    checked_measure(api,work_for('fresh'),fresh,ref,fingerprint,error,'cuda')
    with pytest.raises(ValueError,match='captured output mismatch'):
        checked_measure(api,work_for('stale'),fresh,ref,fingerprint,error,'cuda')
    with pytest.raises(ValueError,match='modified inputs'):
        checked_measure(api,work_for('mutating'),fresh,ref,fingerprint,error,'cuda')
    late=work_for('late')
    checked_measure(api,late,fresh,ref,fingerprint,error,'cuda')
    with pytest.raises(ValueError,match='captured output mismatch'):
        checked_measure(api,late,fresh,ref,fingerprint,error,'cuda')


def test_capture_rejects_shared_cached_output_storage():
    api,_=fake_torch();buffer=torch.ones(3)
    with pytest.raises(GraphCaptureError,match='reuse output storage'):
        capture_work(api,lambda x:buffer,[torch.ones(3)],'cuda')


def test_retained_outputs_detect_early_wrong_last_correct_capture_branch():
    api,_=fake_torch();count=[0]
    def switch(x):
        count[0]+=1
        return x*2 if count[0] in (1,65) else torch.zeros_like(x)
    work=capture_work(api,switch,[torch.ones(3)],'cuda')
    # Last output is correct, but an earlier captured invocation is not.
    assert torch.equal(work.outputs[-1],torch.full((3,),2.))
    from kernel_env.server.cuda_graph_timing import finish_check
    fingerprint=lambda xs:tuple(tuple(x.tolist()) for x in xs)
    before=fingerprint([x for b in work.batches for x in b])
    with pytest.raises(ValueError,match='captured output mismatch'):
        finish_check(api,work,[torch.full((3,),2.) for _ in work.batches],before,fingerprint,
                     lambda a,b:None if torch.equal(a,b) else 'values differ')


def test_graph_incompatible_code_is_explicitly_rejected_without_fallback():
    api,_=fake_torch()
    @contextlib.contextmanager
    def incompatible(*args,**kwargs):
        raise RuntimeError('operation not permitted when stream is capturing')
        yield
    api.cuda.graph=incompatible
    with pytest.raises(GraphCaptureError,match='no alternate timer'):
        capture_work(api,lambda x:x.clone(),[torch.ones(3)],'cuda')


def test_every_graph_warmup_is_validated_before_any_timed_event():
    api,events=fake_torch();work=work_for('late');counter=[0]
    def invalid_warmup():
        counter[0]+=1
        for batch,out in zip(work.batches,work.outputs):out.copy_(batch[0]*2 if counter[0]==1 else torch.zeros(3))
    work.graph=SimpleNamespace(replay=invalid_warmup)
    fingerprint=lambda xs:tuple(tuple(x.tolist()) for x in xs)
    with pytest.raises(ValueError,match='captured output mismatch'):
        checked_measure(api,work,[[torch.ones(3)],[torch.ones(3)]],lambda x:x*2,fingerprint,
                        lambda a,b:None if torch.equal(a,b) else 'wrong','cuda')
    assert counter[0]==2 and 'start' not in events


def test_null_calibration_uses_fixed_warmups_and_retains_all_samples():
    from kernel_env.server.cuda_graph_timing import calibrated_floor,VALIDATED_GRAPH_WARMUPS
    api,events=fake_torch();work=work_for('fresh')
    floor,metadata=calibrated_floor(api,work,'cuda')
    assert metadata['warmup_replays']==VALIDATED_GRAPH_WARMUPS==3
    assert len(metadata['null_graph_per_invocation_ms'])==7
    assert events[:events.index('start')].count('gpu_graph_replay')==3
    assert floor>0


def test_eager_warmup_rejects_wrong_output_and_mutation():
    from kernel_env.server.cuda_graph_timing import checked_eager_warmup
    api,_=fake_torch();fingerprint=lambda xs:tuple(tuple(x.tolist()) for x in xs)
    error=lambda a,b:None if torch.equal(a,b) else 'wrong'
    checked_eager_warmup(api,lambda x:x*2,[torch.ones(3)],lambda x:x*2,fingerprint,error)
    with pytest.raises(ValueError,match='captured output mismatch'):
        checked_eager_warmup(api,lambda x:torch.zeros_like(x),[torch.ones(3)],lambda x:x*2,fingerprint,error)
    def mutates(x):x.zero_();return x.clone()
    with pytest.raises(ValueError,match='warmup modified inputs'):
        checked_eager_warmup(api,mutates,[torch.ones(3)],lambda x:x*2,fingerprint,error)


@pytest.mark.parametrize('mib,repeats',[(2,64),(8,64),(32,16),(64,8),(128,4),(256,2),(512,1),(6144,1)])
def test_fixed_input_storage_target_selects_repeats(mib,repeats):
    from kernel_env.server.cuda_graph_timing import graph_repeats
    assert graph_repeats(mib*1024**2)==repeats


def test_large_control_memory_plan_includes_outputs_workspace_scratch_and_reserve():
    from kernel_env.server.cuda_graph_timing import graph_memory_plan,LIVE_MEMORY_CAP_BYTES
    gib=1024**3
    plan=graph_memory_plan(6*gib,6*gib,1)
    assert plan['estimated_live_bytes']==48.25*gib
    assert plan['estimated_live_bytes']+plan['reserve_bytes']==56.25*gib<=LIVE_MEMORY_CAP_BYTES
    with pytest.raises(ValueError,match='memory estimate exceeds'):
        graph_memory_plan(6*gib,12*gib,1)
    with pytest.raises(ValueError,match='memory estimate exceeds'):
        graph_memory_plan(6*gib,6*gib,64)


def test_actual_memory_peak_is_recorded_and_cap_cannot_ignore_reserved_memory():
    from kernel_env.server.cuda_graph_timing import cuda_memory_evidence
    api,_=fake_torch();gib=1024**3
    api.cuda.max_memory_allocated=lambda device:12*gib
    api.cuda.max_memory_reserved=lambda device:50*gib
    assert cuda_memory_evidence(api,'cuda')['peak_reserved_bytes']==50*gib
    api.cuda.max_memory_reserved=lambda device:57*gib
    with pytest.raises(ValueError,match='actual CUDA memory peak exceeds'):
        cuda_memory_evidence(api,'cuda')


def test_fresh_reference_and_copy_stream_is_ready_before_first_graph_replay():
    api,events=fake_torch();work=work_for('fresh')
    work.stream=SimpleNamespace(wait_stream=lambda stream:events.append('ready_wait'))
    original=work.graph.replay
    work.graph=SimpleNamespace(replay=lambda:(events.append('checked_replay'),original()))
    def ref(x):events.append('fresh_reference');return x*2
    fingerprint=lambda xs:tuple(tuple(x.tolist()) for x in xs)
    checked_measure(api,work,[[torch.ones(3)],[torch.ones(3)]],ref,fingerprint,
                    lambda a,b:None if torch.equal(a,b) else 'wrong','cuda')
    assert events.index('fresh_reference')<events.index('ready_wait')<events.index('checked_replay')


def test_method_and_floor_metadata_describe_unsubtracted_event_interval():
    from kernel_env.server.cuda_graph_timing import METHOD,MEASUREMENT_SCOPE,calibrated_floor
    api,_=fake_torch();floor,metadata=calibrated_floor(api,work_for('fresh'),'cuda')
    assert METHOD=='fresh_paired_cuda_graph_event_interval_v4'
    assert 'unsubtracted launch overhead' in MEASUREMENT_SCOPE
    assert metadata['measurement_scope']==MEASUREMENT_SCOPE
    assert floor==max(metadata['event_resolution_per_invocation_ms'],
                      metadata['null_median_per_invocation_ms'],metadata['null_range_per_invocation_ms'])

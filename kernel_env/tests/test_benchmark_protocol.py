import pytest

from kernel_env.server.benchmark_protocol import paired_timings, cuda_coverage
from types import SimpleNamespace


def test_fresh_paired_timings_randomize_order_and_include_both_uncertainties():
    order=[]
    def ref():order.append('ref');return 10.
    def candidate():order.append('candidate');return 5.
    result=paired_timings(ref,candidate,seed=1)
    assert order.count('ref') == order.count('candidate') == 3
    assert all(set(order[i:i+2]) == {'ref','candidate'} for i in (0,2,4))
    assert result['speedup'] == 2
    assert result['speedup_lower_bound'] == pytest.approx(9.8/5.1)
    assert result['speedup_upper_bound'] == pytest.approx(10.2/4.9)


def test_near_parity_and_unstable_measurements_do_not_claim_clear_speedup():
    result=paired_timings(lambda:4.27,lambda:4.26)
    assert result['speedup_lower_bound'] < 1 < result['speedup_upper_bound']
    values=iter([3.,5.,4.])
    result=paired_timings(lambda:4.27,lambda:next(values))
    assert result['speedup_lower_bound'] < 1


@pytest.mark.parametrize('invalid',[0,float('nan'),float('inf'),-1])
def test_invalid_timing_fails_closed(invalid):
    with pytest.raises(ValueError):paired_timings(lambda:invalid,lambda:1.)


def test_coverage_counts_actual_cuda_events_and_rejects_invalid_durations():
    events=[SimpleNamespace(device_type='DeviceType.CUDA',device_time_total=3,name='relu_kernel',device_index=0,device_resource_id=7),
            SimpleNamespace(device_type='DeviceType.CUDA',device_time_total=1,name='copy',device_index=0,device_resource_id=7),
            SimpleNamespace(device_type='DeviceType.CPU',device_time_total=100,name='relu_kernel')]
    result=cuda_coverage(events,{'relu_kernel'})
    assert result['generated_cuda_fraction']==.75
    assert result['cuda_time_us']==4
    assert result['cuda_streams']==[(0,7)]
    events[0].device_time_total=float('nan')
    with pytest.raises(ValueError):cuda_coverage(events,{'relu_kernel'})


def test_generated_work_does_not_hide_an_unjoined_side_stream():
    events=[SimpleNamespace(device_type='DeviceType.CUDA',device_time_total=8,name='relu_kernel',device_index=0,device_resource_id=7),
            SimpleNamespace(device_type='DeviceType.CUDA',device_time_total=51,name='spin_kernel',device_index=0,device_resource_id=8)]
    with pytest.raises(ValueError,match='side-stream'):
        cuda_coverage(events,{'relu_kernel'})


@pytest.mark.parametrize('device,stream',[(None,7),(0,None),(0,-1)])
def test_missing_stream_evidence_cannot_clear_timing(device,stream):
    event=SimpleNamespace(device_type='DeviceType.CUDA',device_time_total=8,name='relu_kernel',device_index=device,device_resource_id=stream)
    with pytest.raises(ValueError,match='stream evidence missing'):
        cuda_coverage([event],{'relu_kernel'})


def test_calibrated_graph_floor_preserves_resolvable_microsecond_gain():
    result=paired_timings(lambda:.002,lambda:.001,absolute_floor_ms=.00005)
    assert result['speedup_lower_bound']>1.
    noisy=paired_timings(lambda:.002,lambda:.001,absolute_floor_ms=.001)
    assert noisy['speedup_lower_bound']<1<noisy['speedup_upper_bound']

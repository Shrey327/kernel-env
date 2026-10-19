import copy
import pytest
from reward_noise import noise_aware_returns, timing_reward_band


def ep(runtime=4.28,code='x=1',correct=True,mode='roofline'):
    frac=3.84/runtime
    reward=0.3+2.7*min(frac,1) if correct else 0
    return {'task_id':'relu','codes':[code],'turns':[{'correct':correct,'reward':reward,
        'reward_mode':mode,'t_new_ms':runtime,'timing_samples_ms':[runtime-.0001,runtime,runtime+.0001],
        'roofline_frac':frac,'speedup':4.3/runtime}]}


def test_identical_source_and_nearby_kernels_have_zero_false_preference():
    rows=[ep(4.2785),ep(4.2808),ep(4.279,code='y=2'),ep(4.282,code='z=3')]
    rewards, stats=noise_aware_returns(rows,4)
    assert len(set(rewards))==1 and stats['learning_return_std']==0 and stats['raw_return_std']>0
    assert stats['tied_episodes']==4 and len({r['raw_return'] for r in rows})>1


def test_identical_ast_stays_tied_despite_large_clock_drift():
    rows=[ep(4,code='x = 1 # first'),ep(5,code='x=1 # same')]
    rewards,_=noise_aware_returns(rows,2)
    assert rewards[0]==rewards[1]


def test_meaningful_speed_and_correctness_differences_are_preserved():
    rows=[ep(4,code='x=1'),ep(8,code='x=2'),ep(correct=False),ep(4.001,code='x=3')]
    rewards,_=noise_aware_returns(rows,4)
    assert rewards[0]==rewards[3] and rewards[0]>rewards[1]>rewards[2]==0


def test_observed_instability_and_absolute_microsecond_floor_expand_ties():
    a,b=ep(4,code='x=1'),ep(4.5,code='x=2')
    a['turns'][0]['timing_samples_ms']=[3.8,4,4.4]
    assert len(set(noise_aware_returns([a,b],2)[0]))==1
    a,b=ep(.002,code='x=1'),ep(.0028,code='x=2')
    for e in [a,b]:
        t=e['turns'][0];t.update(roofline_frac=.001/t['t_new_ms'],reward=.3+2.7*.001/t['t_new_ms'],timing_samples_ms=[t['t_new_ms']]*3)
    assert len(set(noise_aware_returns([a,b],2)[0]))==1


def test_repair_order_and_task_boundaries_are_not_merged():
    correct,bad=ep()['turns'][0],ep(correct=False)['turns'][0]
    a={'task_id':'relu','codes':['x=1','x=1'],'turns':[correct,bad]}
    b={'task_id':'relu','codes':['x=1','x=1'],'turns':[bad,correct]}
    rewards,_=noise_aware_returns([a,b],2)
    assert rewards[0]>rewards[1]
    b=copy.deepcopy(a);b['task_id']='other'
    with pytest.raises(ValueError):noise_aware_returns([a,b],2)


def test_speedup_fallback_and_invalid_evidence_fail_closed():
    t=ep()['turns'][0];t.update(reward_mode='speedup',speedup=1,reward=.3)
    lo,hi=timing_reward_band(t);assert lo<.3<hi
    t['t_new_ms']=None
    with pytest.raises(ValueError):timing_reward_band(t)
    with pytest.raises(ValueError):noise_aware_returns([ep()],4)

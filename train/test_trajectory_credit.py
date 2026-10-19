import pytest

from trajectory_credit import independent_returns, loo_scale


def test_loo_matches_leave_other_samples_out_without_reward_standardization():
    values=[0.,1.,2.,4.]
    mean=sum(values)/len(values)
    got=[(x-mean)*loo_scale(len(values)) for x in values]
    want=[x-sum(y for j,y in enumerate(values) if j!=i)/3 for i,x in enumerate(values)]
    assert got==pytest.approx(want)
    assert sum(got)==pytest.approx(0)
    with pytest.raises(ValueError):loo_scale(1)


def test_independent_discounted_returns_do_not_mix_timing_preferences_across_samples():
    episodes=[{'task_id':'a','turns':[{'reward':r},{'reward':2}]} for r in (0,1,2,3)]
    got,_=independent_returns(episodes,4,.4)
    assert got==pytest.approx([.8,1.8,2.8,3.8])
    assert all(e['raw_return']==e['learning_return'] for e in episodes)
    episodes[0]['turns'][0]['stage']='infrastructure'
    with pytest.raises(ValueError):independent_returns(episodes,4,.4)

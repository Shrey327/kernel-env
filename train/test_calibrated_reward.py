import itertools,math
import pytest
from kernel_env.server.calibrated_reward import quality,progress_credit,loo_advantages
from trajectory_credit import calibrated_progress_returns


def test_invalid_outputs_never_receive_reward_and_speed_is_bounded_monotone():
    assert quality(False)==quality(True,1,2,valid=False)==0
    points=[quality(True,s*.98,s*1.02) for s in (.2,.5,1,1.1,2,4,8)]
    assert points==sorted(points)
    assert points[0]==points[1]==points[2]==.25
    assert points[3]>.25 and points[4]-points[2]>.9
    assert points[-1]==2.25
    assert all(math.isfinite(x) for x in points)
    with pytest.raises(ValueError):quality(True,float('nan'),2)


def test_noisy_tie_has_no_fake_performance_bonus():
    assert quality(True,.8,1.2)==.25
    assert quality(True,1.05,1.15)>.25


def test_repetition_cannot_farm_correctness_and_best_within_budget_is_order_invariant():
    q=quality(True,2,2)
    assert progress_credit([q,q,q])==(q,[q,0,0])
    assert progress_credit([0,0,q])[0]==q
    values=[.25,.5,1.25]
    value,credits=progress_credit(values)
    assert credits==[.25,.25,.75]
    assert value==pytest.approx(1.25)
    assert value<=max(values)


def test_identical_groups_have_zero_finite_advantages_and_independent_returns():
    assert loo_advantages([.25]*4)==[0]*4
    eps=[{'task_id':'a','turns':[{'stage':'timed','reward':.25}]*3} for _ in range(4)]
    returns,_=calibrated_progress_returns(eps,4,1.)
    assert returns==[.25]*4
    assert eps[0]['per_turn_incremental_credit']==[.25,0,0]


def test_trajectory_loo_score_estimator_matches_independent_reward_expectation():
    # Exhaustive independent Bernoulli4 proof atp=.5, not a noisy statistical test.
    expectation=0
    for group in itertools.product((0.,1.),repeat=4):
        advantages=loo_advantages(group)
        expectation+=sum(a*(x-.5) for a,x in zip(advantages,group))/4/16
        centered=[(x-sum(group)/4)*4/3 for x in group]
        assert centered==pytest.approx(advantages)
    assert expectation==pytest.approx(.25)


@pytest.mark.parametrize('bounds',[(None,None),(-.1,1),(float('inf'),2),(2,1),(1,float('nan'))])
def test_malformed_bounds_fail_closed(bounds):
    with pytest.raises(ValueError):quality(True,*bounds)


def test_wider_uncertainty_never_increases_reward_at_fixed_lower_bound():
    assert quality(True,.4,.99)==quality(True,.4,1.01)==.25


def test_loo_numerical_cases_and_translation_invariance():
    assert loo_advantages([0,1,2,4])==pytest.approx([-7/3,-1,1/3,3])
    assert loo_advantages([2,2,2,2.004])==pytest.approx([-.004/3]*3+[.004])
    assert loo_advantages([10,11,12,14])==pytest.approx(loo_advantages([0,1,2,4]))


@pytest.mark.parametrize('failure',['mixed','incomplete','infra'])
def test_invalid_groups_abort_instead_of_entering_credit(failure):
    episodes=[{'task_id':'a','turns':[{'stage':'timed','reward':.25}]} for _ in range(4)]
    if failure=='mixed':episodes[0]['task_id']='b'
    elif failure=='incomplete':episodes.pop()
    else:episodes[0]['turns'][0]['stage']='infrastructure'
    with pytest.raises(ValueError):calibrated_progress_returns(episodes,4,1.)


def test_actual_return_path_requires_gamma1_and_is_order_invariant():
    early=[{'task_id':'a','turns':[{'reward':r} for r in (1.25,0,0)]} for _ in range(4)]
    late=[{'task_id':'a','turns':[{'reward':r} for r in (0,0,1.25)]} for _ in range(4)]
    assert calibrated_progress_returns(early,4,1.)[0]==calibrated_progress_returns(late,4,1.)[0]==[1.25]*4
    with pytest.raises(ValueError,match='requires gamma1'):calibrated_progress_returns(late,4,.9)


def test_partial_credit_cannot_reward_invalid_in_calibrated_mode(monkeypatch):
    from kernel_env.server.rubrics import kernel_rubric
    from types import SimpleNamespace
    monkeypatch.setenv('KERNEL_ENV_PARTIAL_CREDIT','1')
    for explicit in (None,True,False):
        rubric=kernel_rubric('calibrated_speedup_v2',explicit)
        for stage in ('runtime','timeout','correctness','format'):
            assert rubric.forward(None,SimpleNamespace(correct=False,stage=stage))==0


def calibration_control(reference=.1,floor=.001):
    from kernel_env.server.benchmark_protocol import paired_timings
    replicate=paired_timings(lambda:reference,lambda:reference,absolute_floor_ms=floor)
    return {'replicates':[dict(replicate) for _ in range(3)],'calibration':{'absolute_floor_ms':floor},'input_shape':[32,262144],'input_bytes':32*1024**2}


def test_calibration_retains_later_regimes_after_early_sensitivity_failure():
    from reward_calibration_artifact import calibrate
    value=calibrate({'small':calibration_control(.003344,.0001685),'training':calibration_control()}, {'passed':True})
    assert not value['passed']
    assert len(value['regimes'])==6
    assert all(value['regimes']['training/replicate-'+str(i)]['passed'] for i in (1,2,3))
    assert len(value['regimes']['small/replicate-3']['cases'])==13
    assert value['failures']


def test_calibration_checks_later_replicates_and_missing_required_regime():
    from reward_calibration_artifact import calibrate
    control=calibration_control()
    control['replicates'][2]['ref_timing_samples_ms']=[.003344]*3
    control['calibration']['absolute_floor_ms']=.0001685
    value=calibrate({'training':control},{'passed':True},required_regimes=['training','missing'])
    assert not value['passed'] and len(value['regimes'])==3
    assert value['regimes']['training/replicate-1']['passed']
    assert not value['regimes']['training/replicate-3']['passed']
    assert any(row['regime']=='missing' for row in value['failures'])


def test_small_diagnostic_cannot_block_training_sensitivity_gate():
    from reward_calibration_artifact import calibrate
    value=calibrate({'small':calibration_control(.003344,.0001685),'training':calibration_control()}, {'passed':True},required_regimes=['training'])
    assert value['passed'] and not value['failures'] and value['diagnostic_failures']
    assert not value['regimes']['small/replicate-1']['required_for_training']


def test_calibration_trainer_failure_never_passes_and_incomplete_replicates_fail():
    from reward_calibration_artifact import calibrate
    control=calibration_control();control['replicates'].pop()
    value=calibrate({'training':control},{'passed':False})
    assert not value['passed'] and len(value['regimes'])==2
    assert len(value['failures'])==2

import pytest
from campaign import choose_curriculum, compare_paired, CANDIDATES, RESERVED, EXCLUDED


def episode(task,seed,correct):
    return {'task_id':task,'sample_seed':seed,'turns':[
        {'stage':'timed' if correct else 'correctness','correct':correct,'speedup':1.0 if correct else 0.0}]}


def test_curriculum_never_uses_reserved_or_zero_reward_tasks():
    rows=[episode(CANDIDATES[0],1,True),episode(CANDIDATES[1],1,False)]
    assert choose_curriculum(rows,CANDIDATES)==[CANDIDATES[0]]
    with pytest.raises(ValueError):choose_curriculum([episode(RESERVED[0],1,True)],CANDIDATES)
    assert not set(CANDIDATES)&set(RESERVED) and not set(CANDIDATES+RESERVED)&EXCLUDED


def test_paired_comparison_keeps_failures_and_matches_seeds():
    before=[episode('a',1,False),episode('a',2,True),episode('b',1,False),episode('b',2,False)]
    after=[episode('a',1,True),episode('a',2,True),episode('b',1,False),episode('b',2,False)]
    result=compare_paired(before,after,{'test':['a','b']})['test']
    assert result['paired_correctness_delta_pp']==25
    assert result['before']['correct_pct']==25 and result['after']['correct_pct']==50
    assert result['task_bootstrap_95pct_delta_pp']==[0.0,50.0]
    after[-1]['sample_seed']=3
    with pytest.raises(ValueError):compare_paired(before,after,{'test':['a','b']})


def test_infrastructure_failure_is_not_a_model_score():
    a=episode('a',1,False);a['error']='connection lost'
    with pytest.raises(ValueError):compare_paired([a],[a],{'x':['a']})

from corrected_campaign import completion,pilot_gate
import pytest


def ep(lower,upper,reward,correct=True):
    return {'task_id':'a','turns':[{'stage':'timed','correct':correct,'speedup_lower_bound':lower,
            'speedup_upper_bound':upper,'reward':reward}]}


def test_partial_steps_are_never_reported_complete():
    assert completion(31,32)=={'completed':False,'actual_steps':31,'target_steps':32}
    assert completion(32,32)['completed']


def test_pilot_requires_real_speed_separation_not_only_correctness_or_noise():
    metrics=[{'loss':.1,'grad_norm':.01}]
    assert not pilot_gate([ep(2,2.2,1.25),ep(2,2.2,1.25),ep(0,0,0,False),ep(0,0,0,False)],metrics,.01)['passed']
    assert not pilot_gate([ep(2,2.2,1.25)]*4,metrics,.01)['passed']
    assert pilot_gate([ep(2,2.2,1.25),ep(.9,1.1,.25),ep(0,0,0,False),ep(0,0,0,False)],metrics,.01)['passed']


@pytest.mark.parametrize('noise',[None,-1,float('nan'),float('inf')])
def test_missing_or_invalid_measured_noise_cannot_default_to_zero(noise):
    with pytest.raises(ValueError,match='measured finite'):
        pilot_gate([ep(2,2.2,1.25)]*4,[{'loss':.1,'grad_norm':.01}],noise)


def test_pilot_groups_follow_the_protocol_group_size():
    metrics=[{'loss':.1,'grad_norm':.01}]
    group=[ep(2,2.2,1.25),ep(.9,1.1,.25)]+[ep(0,0,0,False)]*6
    assert pilot_gate(group,metrics,.01,group_size=8)['passed']
    with pytest.raises(ValueError,match='incomplete or mixed'):pilot_gate(group[:6],metrics,.01,group_size=8)


def test_train_command_uses_whole_groups_of_eight(monkeypatch):
    monkeypatch.setenv('KERNEL_ENV_URL','http://127.0.0.1:8000')
    from corrected_campaign import train_command
    cmd=train_command('base','tasks.json','out',4,0.,'contract.json',generations=8,episodes=16)
    assert cmd[cmd.index('--num_generations')+1]=='8' and cmd[cmd.index('--per_device_train_batch_size')+1]=='4'
    assert cmd[cmd.index('--gradient_accumulation_steps')+1]=='4'
    with pytest.raises(ValueError):train_command('base','t','o',4,0.,'c',generations=8,episodes=12)

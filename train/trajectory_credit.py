"""Episode-level LOO for fixed-size groups; not a turn-level TRLOO implementation."""
import math


def loo_scale(group_size):
    if group_size < 2:
        raise ValueError('LOO needs at least two independent trajectories')
    return group_size/(group_size-1)


def independent_returns(episodes, group_size, gamma):
    if not episodes or len(episodes)%group_size or not 0<=gamma<=1:
        raise ValueError('incomplete trajectory groups or invalid discount')
    returns=[]
    for start in range(0,len(episodes),group_size):
        if len({e['task_id'] for e in episodes[start:start+group_size]}) != 1:
            raise ValueError('trajectory group mixes tasks')
    for episode in episodes:
        if any(t.get('stage')=='infrastructure' for t in episode['turns']):
            raise ValueError('infrastructure failure cannot enter policy credit')
        value=sum(gamma**i*float(t['reward']) for i,t in enumerate(episode['turns']))
        if not math.isfinite(value):raise ValueError('invalid trajectory return')
        episode.update(raw_return=value,learning_return=value,return_band=[value,value])
        returns.append(value)
    import statistics
    spread=statistics.pstdev(returns)
    return returns,{'tied_episodes':0,'raw_return_std':spread,'learning_return_std':spread}


def calibrated_progress_returns(episodes,group_size,gamma=1.0):
    from kernel_env.server.calibrated_reward import progress_credit
    if gamma!=1.:raise ValueError('calibrated best-validated objective requires gamma1')
    # Reuse group/infra validation; raw returns remain available as a diagnostic only.
    raw,_=independent_returns(episodes,group_size,gamma)
    values=[]
    for ep in episodes:
        value,credits=progress_credit([float(t['reward']) for t in ep['turns']],gamma)
        ep.update(learning_return=value,per_turn_incremental_credit=credits,return_policy='best-quality-progress-v2')
        values.append(value)
    import statistics
    return values,{'tied_episodes':0,'raw_return_std':statistics.pstdev(raw),'learning_return_std':statistics.pstdev(values)}

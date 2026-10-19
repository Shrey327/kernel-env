"""Conservative timing-equivalence ties for the rewards actually used by GRPO.

The bands are engineering tolerances, not confidence intervals. Correctness patterns
stay separate; overlapping speed bands form conservative connected tie groups.
"""
import ast
import math
import statistics

from baseline_eval import episode_return

RELATIVE_FLOOR = 0.02
ABSOLUTE_FLOOR_MS = 0.001


def timing_reward_band(turn):
    reward = float(turn['reward'])
    if not math.isfinite(reward) or reward < 0:
        raise ValueError('invalid reward')
    if not turn['correct']:
        return reward, reward
    runtime = turn.get('t_new_ms')
    samples = turn.get('timing_samples_ms', [])
    if not runtime or not math.isfinite(runtime) or runtime <= 0 or any(
            not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('correct submission is missing valid timing evidence')
    uncertainty = max(RELATIVE_FLOOR*runtime, ABSOLUTE_FLOOR_MS,
                      max(samples)-min(samples) if samples else 0)
    low_time, high_time = max(runtime-uncertainty, runtime*1e-6), runtime+uncertainty
    fraction = turn.get('roofline_frac')
    if turn.get('reward_mode') == 'roofline' and fraction:
        score = lambda t: 0.3+2.7*min(fraction*runtime/t, 1)
    else:
        speedup = turn.get('speedup')
        if not speedup or not math.isfinite(speedup) or speedup <= 0:
            raise ValueError('correct submission is missing a valid speedup')
        score = lambda t: 0.3+min(max(math.log2(speedup*runtime/t), -0.2), 3)
    if not math.isclose(score(runtime), reward, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('reward policy and timing evidence disagree')
    return score(high_time), score(low_time)


def _source_key(episode):
    keys = []
    for code in episode.get('codes', []):
        try:keys.append(ast.dump(ast.parse(code), include_attributes=False))
        except SyntaxError:keys.append(code)
    return tuple(keys) if len(keys) == len(episode['turns']) else None


def noise_aware_returns(episodes, group_size, mode='discounted', gamma=0.4):
    if not episodes or group_size <= 0 or len(episodes) % group_size:
        raise ValueError('incomplete GRPO generation group')
    if mode not in ('discounted','best') or not 0 <= gamma <= 1:
        raise ValueError('invalid episode return policy')
    raw, bands = [], []
    for ep in episodes:
        raw.append(episode_return([t['reward'] for t in ep['turns']], mode, gamma))
        ranges = [timing_reward_band(t) for t in ep['turns']]
        bands.append([episode_return([b[j] for b in ranges], mode, gamma) for j in (0,1)])
    effective, tied = list(raw), 0
    patterns = [tuple(t['correct'] for t in ep['turns']) for ep in episodes]
    sources = [_source_key(ep) for ep in episodes]
    for start in range(0, len(episodes), group_size):
        group = range(start, start+group_size)
        if len({episodes[i]['task_id'] for i in group}) != 1:
            raise ValueError('GRPO group contains different tasks')
        remaining = set(group)
        while remaining:
            component = {remaining.pop()}
            while True:
                linked = {j for j in remaining if any(
                    patterns[i] == patterns[j]
                    and (max(bands[i][0], bands[j][0]) <= min(bands[i][1], bands[j][1])
                         or (sources[i] is not None and sources[i] == sources[j]))
                    for i in component)}
                if not linked:break
                component.update(linked);remaining.difference_update(linked)
            value = statistics.median(raw[i] for i in component)
            if len(component)>1 and max(raw[i] for i in component)-min(raw[i] for i in component)>1e-12:
                tied += len(component)
            for i in component:effective[i] = value
    for i, ep in enumerate(episodes):
        ep.update(raw_return=raw[i], learning_return=effective[i], return_band=bands[i])
    return effective, {'tied_episodes':tied,'raw_return_std':statistics.pstdev(raw),
                       'learning_return_std':statistics.pstdev(effective)}

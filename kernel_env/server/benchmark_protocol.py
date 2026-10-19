"""Fresh paired timing with engineering uncertainty bounds, not confidence intervals."""
import json
import math
import random
import statistics


def cuda_coverage(events, kernel_names):
    rows=[];streams=set()
    for event in events:
        if not str(event.device_type).endswith('CUDA'):
            continue
        micros=float(event.device_time_total)
        if not math.isfinite(micros) or micros < 0:
            raise ValueError('invalid CUDA profiler duration')
        device=getattr(event,'device_index',None);stream=getattr(event,'device_resource_id',None)
        if not isinstance(device,int) or device<0 or not isinstance(stream,int) or stream<0:
            raise ValueError('CUDA stream evidence missing; cannot validate candidate timing')
        streams.add((device,stream))
        rows.append((event.name,micros,any(name in event.name for name in kernel_names)))
    if len(streams)>1:
        raise ValueError('Candidate CUDA work must use one stream; side-stream work is unsupported')
    total=sum(micros for _,micros,_ in rows)
    custom=sum(micros for _,micros,generated in rows if generated)
    return {'generated_cuda_fraction':custom/total if total>0 else 0.,
            'cuda_time_us':total,'generated_cuda_time_us':custom,'cuda_streams':sorted(streams),
            'top_kernels':sorted(rows,key=lambda row:row[1],reverse=True)[:3]}


def generated_replay_profile(run, attempts=3):
    """Replay of a captured graph is fixed work, so only a trace with no CUDA events at all
    (a dropped profiler record) is retried; real but non-generated work is rejected at once."""
    for attempt in range(1, attempts+1):
        summary=run();summary['profile_attempts']=attempt
        if summary['generated_cuda_fraction'] > 0:
            return summary
        if summary['cuda_time_us'] > 0 or summary['top_kernels']:
            break
    raise ValueError('No generated CUDA work in captured replay: '+json.dumps(summary,default=str)[:400])

PAIRED_ROUNDS=7


def interquartile_range(values):
    # A single interference spike widens the full range of 3 samples; the IQR of 7 ignores it.
    q1,_,q3=statistics.quantiles(values,n=4,method='inclusive')
    return q3-q1


def paired_timings(measure_reference, measure_candidate, seed=0, absolute_floor_ms=.001, rounds=PAIRED_ROUNDS):
    if not math.isfinite(absolute_floor_ms) or absolute_floor_ms<0:raise ValueError("invalid timing floor")
    if rounds<5:raise ValueError('at least five paired rounds required for an interquartile band')
    rng = random.Random(seed)
    reference, candidate = [], []
    for _ in range(rounds):
        order = [('reference', measure_reference), ('candidate', measure_candidate)]
        rng.shuffle(order)
        for name, measure in order:
            value = float(measure())
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid paired timing')
            (reference if name == 'reference' else candidate).append(value)
    ref, new = statistics.median(reference), statistics.median(candidate)
    ref_band = max(.02*ref, absolute_floor_ms, interquartile_range(reference))
    new_band = max(.02*new, absolute_floor_ms, interquartile_range(candidate))
    return {'t_ref_ms':ref, 't_new_ms':new, 'ref_timing_samples_ms':reference,
            'timing_samples_ms':candidate, 'speedup':ref/new,
            'speedup_lower_bound':max(0,ref-ref_band)/(new+new_band),
            'speedup_upper_bound':(ref+ref_band)/max(new-new_band,new*1e-6),
            'timing_method':'fresh_paired_device_synchronized_medians'}

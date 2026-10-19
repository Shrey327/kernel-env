"""Fresh paired timing with engineering uncertainty bounds, not confidence intervals."""
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


def paired_timings(measure_reference, measure_candidate, seed=0, absolute_floor_ms=.001):
    if not math.isfinite(absolute_floor_ms) or absolute_floor_ms<0:raise ValueError("invalid timing floor")
    rng = random.Random(seed)
    reference, candidate = [], []
    for _ in range(3):
        order = [('reference', measure_reference), ('candidate', measure_candidate)]
        rng.shuffle(order)
        for name, measure in order:
            value = float(measure())
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid paired timing')
            (reference if name == 'reference' else candidate).append(value)
    ref, new = statistics.median(reference), statistics.median(candidate)
    ref_band = max(.02*ref, absolute_floor_ms, max(reference)-min(reference))
    new_band = max(.02*new, absolute_floor_ms, max(candidate)-min(candidate))
    return {'t_ref_ms':ref, 't_new_ms':new, 'ref_timing_samples_ms':reference,
            'timing_samples_ms':candidate, 'speedup':ref/new,
            'speedup_lower_bound':max(0,ref-ref_band)/(new+new_band),
            'speedup_upper_bound':(ref+ref_band)/max(new-new_band,new*1e-6),
            'timing_method':'fresh_paired_device_synchronized_medians'}

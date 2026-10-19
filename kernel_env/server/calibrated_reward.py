"""Corrected reward v2: correctness credit plus reliable log-speed improvement."""
import math
SPEC_VERSION='corrected-speed-progress-v2'
CORRECTNESS=.25
PERFORMANCE_CAP=2.0
GAMMA=1.0


def quality(correct,lower=None,upper=None,valid=True):
    if not correct or not valid:return 0.
    if (not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (lower,upper)) or
        lower<0 or upper<lower):raise ValueError('valid paired timing bounds required')
    # No performance bonus for slower, parity or unresolved noisy gains.
    performance=min(PERFORMANCE_CAP,max(0.,math.log2(lower))) if lower>0 else 0.
    return CORRECTNESS+performance


def progress_credit(qualities,gamma=GAMMA):
    if not 0<=gamma<=1:raise ValueError('invalid discount')
    best=0.;credits=[]
    for value in qualities:
        if not math.isfinite(value) or not 0<=value<=CORRECTNESS+PERFORMANCE_CAP:
            raise ValueError('invalid calibrated quality')
        credits.append(max(0.,value-best));best=max(best,value)
    return sum(gamma**i*value for i,value in enumerate(credits)),credits


def loo_advantages(returns):
    if len(returns)<2 or not all(math.isfinite(x) for x in returns):raise ValueError('invalid independent group')
    return [value-(sum(returns)-value)/(len(returns)-1) for value in returns]

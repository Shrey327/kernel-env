"""Exact reward/trajectory/advantage response under empirical timing uncertainty."""
import statistics
from types import SimpleNamespace
from kernel_env.server.benchmark_protocol import paired_timings
from kernel_env.server.calibrated_reward import quality,loo_advantages,SPEC_VERSION
from kernel_env.server.rubrics import kernel_rubric
from trajectory_credit import calibrated_progress_returns


def calibrate(regimes,trainer_proof,required_regimes=None):
    """Retain every fixed replicate, including failures; synthetic speeds are not kernels."""
    rubric=kernel_rubric('calibrated_speedup_v2',False);output={};failures=[];diagnostic_failures=[]
    required=set(regimes if required_regimes is None else required_regimes)
    for missing in sorted(required-set(regimes)):
        failures.append({'regime':missing,'reason':'required identity measurements missing'})
    if not trainer_proof.get('passed'):
        failures.append({'regime':'actual installed trainer','reason':'trainer proof failed or not run'})
    for regime,control in regimes.items():
        required_case=regime in required
        target=failures if required_case else diagnostic_failures
        replicates=control.get('replicates',[])
        if len(replicates)!=3:
            target.append({'regime':regime,'reason':'expected exactly three fixed replicates'})
        for index,replicate in enumerate(replicates):
            name=regime+'/replicate-'+str(index+1);rows=[];checks={};groups={}
            try:
                reference=replicate['ref_timing_samples_ms'];floor=control['calibration']['absolute_floor_ms']
                for label,speed,valid in [('incorrect',1.,False),('cheating',4.,False),('cached',4.,False),('mutation',4.,False),
                                          ('late-switch',4.,False),('graph-unsupported',4.,False),('correct-slower',.5,True),
                                          ('parity',1.,True),('noisy-tie',1.01,True),('reliable1.1x',1.1,True),('2x',2.,True),('4x',4.,True),('8x-cap',8.,True)]:
                    refs=iter(reference);cands=iter([x/speed for x in reference])
                    timed=paired_timings(lambda:next(refs),lambda:next(cands),seed=42,absolute_floor_ms=floor)
                    observation=SimpleNamespace(correct=valid,speedup_lower_bound=timed['speedup_lower_bound'],speedup_upper_bound=timed['speedup_upper_bound'])
                    reward=rubric.forward(None,observation)
                    episode={'task_id':'synthetic/'+name,'turns':[{'stage':'timed' if valid else 'format','reward':reward} for _ in range(3)]}
                    returns,_=calibrated_progress_returns([dict(episode) for _ in range(4)],4,1.)
                    rows.append({'case':label,'raw_speedup':speed,'conservative_speedup':timed['speedup_lower_bound'],
                                 'upper_bound':timed['speedup_upper_bound'],'per_turn_quality':reward,'trajectory_repeated3':returns[0],
                                 'all_identical_group_advantages':loo_advantages(returns)})
                keyed={r['case']:r for r in rows}
                checks['invalid_zero']=all(r['per_turn_quality']==0 for r in rows[:6])
                checks['resolves_reliable1.1x']=keyed['reliable1.1x']['per_turn_quality']>keyed['parity']['per_turn_quality']
                checks['2x_above1.1x']=keyed['2x']['per_turn_quality']>keyed['reliable1.1x']['per_turn_quality']
                checks['4x_above2x']=keyed['4x']['per_turn_quality']>keyed['2x']['per_turn_quality']
                checks['ceiling']=keyed['8x-cap']['per_turn_quality']==2.25
                for labels in [('incorrect','parity','reliable1.1x','2x'),('incorrect','cheating','correct-slower','4x')]:
                    values=[keyed[k]['per_turn_quality'] for k in labels]
                    groups[','.join(labels)]={'returns':values,'loo_advantages':loo_advantages(values)}
                repeat_scores=[quality(True,r['speedup_lower_bound'],r['speedup_upper_bound']) for r in replicates]
                checks['identity_reward_constant']=statistics.pstdev(repeat_scores)==0
                value={'input_shape':control['input_shape'],'input_bytes':control['input_bytes'],'empirical_floor_ms':floor,
                       'cases':rows,'mixed_groups':groups,'identical_code_raw_speedups':[r['speedup'] for r in replicates],
                       'identical_code_rewards':repeat_scores,'identical_code_reward_std':statistics.pstdev(repeat_scores)}
            except Exception as error:
                checks['complete']=False
                value={'cases':rows,'error':type(error).__name__+': '+str(error)}
            failed=[key for key,passed in checks.items() if not passed]
            if failed:target.append({'regime':name,'reason':'failed checks','checks':failed})
            output[name]={**value,'required_for_training':required_case,'passed':not failed,'checks':checks,
                          'note':'Synthetic speed responses using measured identity-control timing samples/floor. These are not achieved model speedups.'}
    return {'passed':not failures,'failures':failures,'diagnostic_failures':diagnostic_failures,
            'required_regimes':sorted(required),'reward_spec':SPEC_VERSION,'objective':'best validated quality within3fixed attempts',
            'quality_formula':'invalid0; correct0.25+clip(log2(conservative_lower_bound),0,2)',
            'trajectory':'sum positive best-quality increments; gamma1 exactly equals max quality; no repeated-correctness bonus',
            'turn_order_tradeoff':'No preference for early versus late identical final quality; attempt budget is fixed. This matches fastest correct kernel within refinement budget.',
            'ceiling':2.25,'ceiling_speedup_conservative':4.,'raw_speedup_is_not_rewarded_speedup':True,
            'calibration_kind':'engineering bounds, not confidence intervals','actual_installed_trainer':trainer_proof,'regimes':output,
            'limitations':['Trajectory-level credit shared across generated tokens, not per-turn TRLOO.','No claim of guaranteed generalization or model improvement.']}

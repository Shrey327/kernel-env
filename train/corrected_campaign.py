"""Corrected staged campaign. No automatic retry, old-run resume or training→evaluation chain."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
from corrected_contract import require_gate
from better_campaign import materialize_model,train_command as old_train_command,run
from campaign import read_episodes,compare_paired


def train_command(base,tasks,out,steps,deadline,contract):
    cmd=old_train_command(base,tasks,out,steps,42,0,deadline)
    cmd[cmd.index('--gamma')+1]='1.0'
    return cmd+['--calibrated_progress_returns','true','--protocol_manifest',str(contract),'--multi_objective_aggregation','sum_then_normalize','--beta','0.0']


def completion(actual,target):return {'completed':actual==target,'actual_steps':actual,'target_steps':target}


def pilot_gate(episodes,metrics,noise_std):
    if not isinstance(noise_std,(int,float)) or not math.isfinite(noise_std) or noise_std<0:
        raise ValueError('measured finite fast-code reward noise is required')
    if any(e.get('error') or any(t.get('stage')=='infrastructure' for t in e['turns']) for e in episodes):
        raise ValueError('pilot infrastructure failure')
    faster=[e for e in episodes if any(t.get('correct') and (t.get('speedup_lower_bound') or 0)>1 for t in e['turns'])]
    separated=0
    for start in range(0,len(episodes),4):
        group=episodes[start:start+4]
        if len(group)!=4 or len({e['task_id'] for e in group})!=1:raise ValueError('pilot incomplete or mixed group')
        correct=[]
        for ep in group:
            good=[t for t in ep['turns'] if t.get('correct')]
            if good:correct.append(max(good,key=lambda t:t.get('reward',0)))
        for i,a in enumerate(correct):
            for b in correct[i+1:]:
                if (a.get('speedup_lower_bound',0)>b.get('speedup_upper_bound',math.inf) or
                    b.get('speedup_lower_bound',0)>a.get('speedup_upper_bound',math.inf)) and abs(a['reward']-b['reward'])>max(.05,3*noise_std):
                    separated+=1
    finite=all(math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) for m in metrics)
    signal=bool(faster and separated and any(m['grad_norm']>0 for m in metrics))
    return {'passed':finite and signal,'correct_faster_episodes':len(faster),'non_noise_performance_pairs':separated,
            'empirical_fast_code_reward_std':noise_std,'finite_optimizer_metrics':finite,
            'reason':'pass' if finite and signal else 'Insufficient demonstrated speed-learning signal; full training held. All correct sub-parity/parity kernels plateau at0.25. Propose more suitable train tasks/exploration or better timer before spending full training.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--contract',required=True)
    p.add_argument('--phase',choices=('pilot','train','eval-base','eval-sft','eval-rl'),required=True);p.add_argument('--deadline',required=True,type=float)
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path(a.contract).read_text());protocol=c['protocol'];gates=json.loads((root/'gates.json').read_text())
    for name in ('gpu-validation','reward-calibration','sft-verification'):require_gate(gates[name],c,name)
    if a.phase=='train':
        require_gate(gates['pilot'],c,'pilot');require_gate(gates['independent-review'],c,'independent-review')
    root.joinpath('experiment/seed42').mkdir(parents=True,exist_ok=True)
    (root/'experiment/manifest.json').write_text(json.dumps({**protocol,'model':protocol['model'],'steps':32,'groups':protocol['evaluation_groups']},indent=2))
    selection=root/'selection.json'
    expected_sft='/runs/'+c['sft_source']
    if not selection.exists():selection.write_text(json.dumps({'warmup':expected_sft}))
    if json.loads(selection.read_text()).get('warmup')!=expected_sft:raise ValueError('SFT selection identity changed')
    if hashlib.sha256((Path(expected_sft)/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()!=c['sft_sha256']:
        raise ValueError('clean SFT weights changed after gate')
    if a.phase in ('pilot','train'):
        steps=4 if a.phase=='pilot' else 32
        target=out/'adapter' if a.phase=='pilot' else root/'experiment/seed42/adapter'
        if target.exists():raise RuntimeError('preserve existing corrected adapter; no automatic resume')
        tasks=out/'train_tasks.json';tasks.write_text(json.dumps(protocol['curriculum']))
        base=materialize_model(root)
        seconds=run(train_command(base,tasks,target,steps,a.deadline,Path(a.contract)),out/'train.log',a.deadline)
        state=json.loads((target/'live_trainer_state.json').read_text());result=completion(state['global_step'],steps)
        result.update(seconds=seconds,source_digest=c['source_digest'],protocol_digest=c['protocol_digest'])
        if a.phase=='pilot':
            episodes=read_episodes(target/'rollouts-rank0.jsonl');metrics=[m for m in state['log_history'] if 'rollout/wall_seconds' in m]
            noise=gates['reward-calibration'].get('fast_code_reward_std')
            result.update(pilot_gate(episodes,metrics,noise));result['passed']=result['passed'] and result['completed']
        else:result['passed']=result['completed']
        (out/'result.json').write_text(json.dumps(result,indent=2));return
    arm=a.phase.removeprefix('eval-')
    if arm=='rl':
        require_gate(gates['train'],c,'train')
        if not gates['train'].get('completed'):raise ValueError('partial RL evaluation requires explicit saved-checkpoint plan')
    from baseline_eval import start_vllm_server
    groups=protocol['evaluation_groups'];ids=sum(groups.values(),[]);base=materialize_model(root,arm)
    server=start_vllm_server(str(base),8001,12288,1,0,'--enable-prefix-caching --enable-chunked-prefill --max-num-seqs8 --gpu-memory-utilization0.80 --generation-config vllm'.replace('--max-num-seqs8','--max-num-seqs 8').replace('--gpu-memory-utilization0.80','--gpu-memory-utilization 0.80'))
    try:
        run(['baseline_eval.py','--env-url',os.environ['KERNEL_ENV_URL'],'--backend','openai','--model',str(base),
             '--max-new-tokens','1536','--samples',str(protocol['evaluation_samples']),'--temperature','.7','--seed','1729',
             '--task-ids',*ids,'--out',str(out/'episodes.jsonl')],out/'evaluate.log',a.deadline)
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except __import__('subprocess').TimeoutExpired:server.kill();server.wait()
    episodes=read_episodes(out/'episodes.jsonl')
    if len(episodes)!=len(ids)*protocol['evaluation_samples'] or any(ep.get('error') or ep.get('infrastructure_error') for ep in episodes):
        raise ValueError('evaluation incomplete or infrastructure contaminated; no valid comparison')
    evaluations=root/'evaluations';evaluations.mkdir(exist_ok=True);dest=evaluations/(arm+'.jsonl')
    if dest.exists():raise RuntimeError('preserve existing evaluation; no duplicate arm')
    dest.write_text((out/'episodes.jsonl').read_text())
    if arm=='base':(root/'experiment/before.jsonl').write_text(dest.read_text())
    if arm=='rl':(root/'experiment/seed42/after.jsonl').write_text(dest.read_text())
    if all((evaluations/(k+'.jsonl')).exists() for k in ('base','sft','rl')):
        data={k:read_episodes(evaluations/(k+'.jsonl')) for k in ('base','sft','rl')}
        comparison={x+'_to_'+y:compare_paired(data[x],data[y],groups) for x,y in [('base','sft'),('sft','rl'),('base','rl')]}
        comparison['protocol_digest']=c['protocol_digest'];comparison['source_digest']=c['source_digest']
        (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
        (root/'experiment/seed42/comparison.json').write_text(json.dumps(comparison['base_to_rl'],indent=2))
    (out/'result.json').write_text(json.dumps({'passed':True,'completed':True,'arm':arm,'episodes':len(episodes)},indent=2))


if __name__=='__main__':main()

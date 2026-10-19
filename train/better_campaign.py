"""Gated new experiment. Every arm uses the same verifier, GPU type and frozen evaluation."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from better_protocol import CURRICULUM
from better_warmup import BASE,REVISION
from campaign import DIAGNOSTIC,RESERVED,EXCLUDED,compare_paired,read_episodes
from kernel_env.server.tasks import load_tasks

EVAL_SAMPLES=1


def frozen_protocol(tasks):
    heldout=sorted(t for t,v in tasks.items() if getattr(v,'level',None)==1 and v.split=='eval' and t not in EXCLUDED)
    groups={'training_diagnostic':DIAGNOSTIC,'reserved_transfer':RESERVED,'official_heldout':heldout}
    assert not set(CURRICULUM)&set(RESERVED+heldout)
    return {'model':BASE,'model_revision':REVISION,'steps':32,'episodes_per_update':8,
        'groups':groups,'candidate_tasks':CURRICULUM,'eval_samples':EVAL_SAMPLES,'eval_seed':1729,
        'eval_temperature':0.7,'max_turns':3,'max_new_tokens':1536,'hardware':'H100 trainer + H100 evaluator',
        'training_shapes':'medium variant; held-out and reserved evaluation use original shapes',
        'protocol':'verified SFT; independent paired-speedup returns; gamma0.4; trajectory LOO; optimizer barrier',
        'budget_cap_usd':100,'prior_reported_usd':None}


def train_command(base,tasks,out,steps,seed,pipeline,deadline,learning_rate=1.5e-5):
    return ['grpo_kernel.py','--model_name_or_path',str(base),'--dtype','bfloat16','--bf16','true',
       '--env_url',os.environ['KERNEL_ENV_URL'],'--task_file',str(tasks),
       '--use_peft','--lora_r','16','--lora_alpha','32','--lora_target_modules','all-linear',
       '--use_vllm','--vllm_mode','colocate','--vllm_gpu_memory_utilization','0.35',
       '--vllm_max_model_length','12288','--num_generations','4','--per_device_train_batch_size','4',
       '--gradient_accumulation_steps','2','--max_completion_length','1536','--max_steps',str(steps),
       '--learning_rate',str(learning_rate),'--gradient_checkpointing','true','--max_grad_norm','0.05',
       '--loss_type','dr_grpo','--scale_rewards','none','--trajectory_loo','true',
       '--independent_timing_returns','true','--return_mode','discounted','--gamma','0.4',
       '--temperature','0.8','--rollout_batch_size',str(pipeline),'--stop_at_unix',str(deadline),
       '--seed',str(seed),'--logging_steps','1','--report_to','none','--save_steps','4',
       '--save_total_limit','2','--output_dir',str(out)]


def run(command,log,deadline):
    if time.time()>deadline-30:raise TimeoutError('phase deadline leaves no execution reserve')
    start=time.monotonic()
    with Path(log).open('w') as f:
        proc=subprocess.Popen([sys.executable,'-u',*command],cwd='/root/train',stdout=f,
          stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
        try:rc=proc.wait(timeout=max(1,deadline-time.time()-15))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise TimeoutError(str(log))
    if rc:raise RuntimeError(f'phase exited {rc}: {log}')
    return time.monotonic()-start



def rl_artifact(root,selection):
    default=root/'experiment/seed42/adapter'
    path=Path(selection.get('rl_checkpoint',str(default))).resolve()
    if path!=default.resolve():
        if path.parent!=default.resolve() or not path.name.startswith('checkpoint-'):
            raise ValueError('RL checkpoint must belong to the new experiment seed42 adapter')
        proof=selection.get('rl_checkpoint_provenance',{})
        if not proof.get('verified_intact') or Path(proof.get('path','')).resolve()!=path:
            raise ValueError('explicit verified checkpoint provenance is required')
        state=json.loads((path/'trainer_state.json').read_text())
        step=state['global_step']
        if path.name!=f'checkpoint-{step}' or proof.get('actual_step')!=step:
            raise ValueError('checkpoint step provenance mismatch')
    else:
        state=json.loads((path/'live_trainer_state.json').read_text());step=state['global_step']
    if not 0<step<=32:raise ValueError('invalid saved RL step')
    if not all((path/name).is_file() for name in ('adapter_model.safetensors','adapter_config.json')):
        raise ValueError('RL adapter files missing')
    return path,step


def materialize_model(root,arm='sft'):
    from huggingface_hub import snapshot_download
    base=snapshot_download(BASE,revision=REVISION)
    if arm=='base':return base
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    import torch
    selection=json.loads((root/'selection.json').read_text())
    adapter=Path(selection['warmup'])/'adapter'
    model=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,adapter).merge_and_unload()
    if arm=='rl':
        adapter,actual_step=rl_artifact(root,selection)
        model=PeftModel.from_pretrained(model,adapter).merge_and_unload()
    path=Path('/tmp')/('better-'+arm);path.mkdir(exist_ok=True)
    model.save_pretrained(path);AutoTokenizer.from_pretrained(base).save_pretrained(path)
    del model
    return path


def pilot_summary(path,seconds):
    state=json.loads((path/'adapter/live_trainer_state.json').read_text())
    metrics=[row for row in state['log_history'] if 'rollout/wall_seconds' in row]
    metric=metrics[-1];episodes=read_episodes(path/'adapter/rollouts-rank0.jsonl')
    if any(ep.get('error') or any(t['stage']=='infrastructure' for t in ep['turns']) for ep in episodes):
        raise RuntimeError('pilot infrastructure failure')
    correct=sum(any(t.get('correct') for t in ep['turns']) for ep in episodes)
    rollout=sum(float(row['rollout/wall_seconds']) for row in metrics)
    generation=sum(float(row['rollout/generation_seconds']) for row in metrics)
    generated=sum(float(row.get('rollout/generated_tokens',row['rollout/tokens_per_generation_second']*row['rollout/generation_seconds'])) for row in metrics)
    return {'wall_seconds':seconds,'rollout_seconds':rollout,'episodes':len(episodes),'correct_episodes':correct,
       'completed_episodes_per_reserved_dollar':len(episodes)/(seconds*10/3600) if seconds else None,
       'correct_episodes_per_reserved_dollar':correct/(seconds*10/3600) if seconds else None,
       'completed_episodes_per_rollout_dollar':len(episodes)/(rollout*10/3600),
       'correct_episodes_per_rollout_dollar':correct/(rollout*10/3600),
       'generation_seconds':generation,
       'generation_tokens_per_second':generated/generation if generation else None,
       'optimizer_updates':len(metrics),
       'step_seconds':metric.get('step_time'),'loss':metric.get('loss'),'grad_norm':metric.get('grad_norm')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['pilot','train','evaluate'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--arm',choices=['base','sft','rl'],default='rl')
    p.add_argument('--reuse-control')
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    tasks=load_tasks();assert all(tasks[t].split=='train' for t in CURRICULUM)
    protocol=frozen_protocol(tasks)
    protocol_path=root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('frozen experimental protocol changed')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    manifest=root/'experiment/manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(protocol,indent=2))
    task_file=out/'train_tasks.json';task_file.write_text(json.dumps(CURRICULUM))
    if a.phase=='pilot':
        base=materialize_model(root);results={}
        for pipeline in (0,4):
            if pipeline==0 and a.reuse_control:
                path=Path(a.reuse_control).resolve()
                if root.resolve() not in path.parents:raise ValueError('control must be in this experiment')
                if json.loads((path.parent/'train_tasks.json').read_text())!=CURRICULUM:raise ValueError('control curriculum mismatch')
                state=json.loads((path/'adapter/live_trainer_state.json').read_text())
                row=next(r for r in reversed(state['log_history']) if 'rollout/wall_seconds' in r)
                if state['global_step']!=1 or row['learning_rate']!=0:raise ValueError('only one-step frozen control may be reused')
                results['0']={**pilot_summary(path,None),'reused_from':str(path)}
                continue
            path=out/f'batch{pipeline}';path.mkdir()
            limit=min(a.deadline-240,time.time()+360)
            seconds=run(train_command(base,task_file,path/'adapter',1,42,pipeline,limit,0.),path/'train.log',limit)
            results[str(pipeline)]=pilot_summary(path,seconds)
        sync,pipeline=results['0'],results['4']
        control_path=Path(a.reuse_control) if a.reuse_control else out/'batch0'
        matched=sorted(e['task_id'] for e in read_episodes(control_path/'adapter/rollouts-rank0.jsonl'))==sorted(e['task_id'] for e in read_episodes(out/'batch4/adapter/rollouts-rank0.jsonl'))
        selected=4 if matched and pipeline['rollout_seconds']<=sync['rollout_seconds']*.9 and pipeline['correct_episodes']>=sync['correct_episodes'] and pipeline['completed_episodes_per_rollout_dollar']>=sync['completed_episodes_per_rollout_dollar']*1.1 else 0
        sanity=out/'sanity';sanity.mkdir()
        seconds=run(train_command(base,task_file,sanity/'adapter',2,42,selected,a.deadline),sanity/'train.log',a.deadline)
        summary=pilot_summary(sanity,seconds)
        if not summary['episodes'] or not math.isfinite(summary['loss']) or not math.isfinite(summary['grad_norm']) or summary['grad_norm']<=0:
            raise RuntimeError('nonfinite or zero-gradient RL pilot; full training blocked')
        result={'passed':True,'pipeline':selected,'matched_task_groups':matched,'measurements':results,'sanity':summary,
          'note':'Small matched-seed pilot on cold separate containers. Rollout-dollar rates exclude model startup/optimizer; full phase wall time is null for the reused historical control. Pipeline retains the optimizer barrier. Serialized evaluator timing is unchanged. This is not evidence of model quality.'}
        (out/'result.json').write_text(json.dumps(result,indent=2))
        selection=json.loads((root/'selection.json').read_text());selection.update(pilot=str(out),pipeline=selected)
        (root/'selection.json').write_text(json.dumps(selection,indent=2))
    elif a.phase=='train':
        selection=json.loads((root/'selection.json').read_text())
        if not json.loads((Path(selection['pilot'])/'result.json').read_text())['passed']:raise RuntimeError('pilot gate missing')
        base=materialize_model(root);seed=root/'experiment/seed42';seed.mkdir(parents=True,exist_ok=True)
        if (seed/'adapter').exists():raise RuntimeError('preserve existing new-run adapter; explicit recovery required')
        seconds=run(train_command(base,task_file,seed/'adapter',32,42,selection['pipeline'],a.deadline),seed/'train.log',a.deadline)
        (out/'result.json').write_text(json.dumps({'completed':True,'seconds':seconds,'seed':42,'target_steps':32,'actual_steps':json.loads((seed/'adapter/live_trainer_state.json').read_text())['global_step']},indent=2))
    else:
        from baseline_eval import start_vllm_server
        groups=protocol['groups']
        ids=sum(groups.values(),[]);assert not set(CURRICULUM)&set(RESERVED+groups['official_heldout'])
        model=materialize_model(root,a.arm)
        provenance={'arm':a.arm,'base':BASE,'base_revision':REVISION,'target_rl_steps':32}
        if a.arm!='base':provenance['sft_adapter']=json.loads((root/'selection.json').read_text())['warmup']+'/adapter'
        if a.arm=='rl':
            adapter,actual_step=rl_artifact(root,json.loads((root/'selection.json').read_text()))
            provenance.update(rl_adapter=str(adapter),actual_rl_steps=actual_step,full_target_completed=actual_step==32)
        (out/'model-provenance.json').write_text(json.dumps(provenance,indent=2))
        server=start_vllm_server(str(model),8001,12288,1,0,
          '--enable-prefix-caching --enable-chunked-prefill --max-num-seqs 8 --gpu-memory-utilization 0.80 --generation-config vllm')
        try:
            run(['baseline_eval.py','--env-url',os.environ['KERNEL_ENV_URL'],'--backend','openai','--model',str(model),
              '--max-new-tokens','1536','--samples',str(EVAL_SAMPLES),'--temperature','0.7','--seed','1729',
              '--task-ids',*ids,'--out',str(out/'episodes.jsonl')],out/'evaluate.log',a.deadline)
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        episodes=read_episodes(out/'episodes.jsonl')
        if len(episodes)!=EVAL_SAMPLES*len(ids) or any(ep.get('error') or ep.get('infrastructure_error') for ep in episodes):
            raise RuntimeError('matched evaluation incomplete or infrastructure-failed')
        frozen=root/'evaluations';frozen.mkdir(exist_ok=True)
        target=frozen/(a.arm+'.jsonl')
        if target.exists():raise RuntimeError('preserve prior evaluation arm')
        target.write_text((out/'episodes.jsonl').read_text())
        if a.arm=='base':(root/'experiment/before.jsonl').write_text(target.read_text())
        if a.arm=='rl':(root/'experiment/seed42/after.jsonl').write_text(target.read_text())
        if all((frozen/(arm+'.jsonl')).exists() for arm in ('base','sft','rl')):
            data={arm:read_episodes(frozen/(arm+'.jsonl')) for arm in ('base','sft','rl')}
            comparison={'base_to_sft':compare_paired(data['base'],data['sft'],groups),
                        'sft_to_rl':compare_paired(data['sft'],data['rl'],groups),
                        'base_to_rl':compare_paired(data['base'],data['rl'],groups)}
            (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
            (root/'experiment/seed42/comparison.json').write_text(json.dumps(comparison['base_to_rl'],indent=2))
        (out/'result.json').write_text(json.dumps({'completed':True,'arm':a.arm,'episodes':len(episodes),'groups':groups},indent=2))


if __name__=='__main__':main()

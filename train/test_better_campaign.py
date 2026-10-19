import json
import pytest
from better_campaign import pilot_summary


def test_pilot_reads_optimizer_row_before_final_training_summary(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir()
    metric={'step':1,'rollout/wall_seconds':20.,'rollout/generation_seconds':3.,
            'rollout/tokens_per_generation_second':10.,'step_time':25.,'loss':-.01,'grad_norm':.02}
    (adapter/'live_trainer_state.json').write_text(json.dumps({'global_step':1,
        'log_history':[metric,{'step':1,'train_runtime':26.,'train_loss':-.01}]}))
    (adapter/'rollouts-rank0.jsonl').write_text(json.dumps({'task_id':'relu','turns':[{'stage':'timed','correct':True}]})+'\n')
    result=pilot_summary(tmp_path,30.)
    assert result['rollout_seconds']==20.
    assert result['grad_norm']==.02
    assert result['completed_episodes_per_reserved_dollar']==pytest.approx(12.)
    reused=pilot_summary(tmp_path,None)
    assert reused['wall_seconds'] is None
    assert reused['completed_episodes_per_reserved_dollar'] is None
    assert reused['completed_episodes_per_rollout_dollar']==pytest.approx(18.)


def test_full_pilot_controller_reuses_control_and_handles_final_summary(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    import better_campaign as module
    root=tmp_path/'experiment';root.mkdir()
    (root/'selection.json').write_text(json.dumps({'warmup':str(root/'sft')}))
    control=root/'pilot/old/batch0';(control/'adapter').mkdir(parents=True)
    (control.parent/'train_tasks.json').write_text(json.dumps(module.CURRICULUM))
    tasks=module.CURRICULUM[:2]
    def records(folder,step,rollout,lr):
        folder.mkdir(parents=True,exist_ok=True)
        metric={'step':step,'learning_rate':lr,'rollout/wall_seconds':rollout,
                'rollout/generation_seconds':3.,'rollout/tokens_per_generation_second':10.,
                'step_time':rollout+5,'loss':-.01,'grad_norm':.02}
        (folder/'live_trainer_state.json').write_text(json.dumps({'global_step':step,
            'log_history':[metric,{'step':step,'train_runtime':rollout+6}]}))
        episodes=[{'task_id':task,'turns':[{'stage':'timed','correct':True}]} for _ in range(step) for task in tasks for i in range(4)]
        (folder/'rollouts-rank0.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    records(control/'adapter',1,20.,0.)
    monkeypatch.setattr(module,'load_tasks',lambda:{t:SimpleNamespace(split='train') for t in module.CURRICULUM})
    monkeypatch.setattr(module,'materialize_model',lambda root:'/tmp/frozen-sft')
    calls=[]
    def run(command,log,deadline):
        calls.append(command)
        target=__import__('pathlib').Path(command[command.index('--output_dir')+1])
        steps=int(command[command.index('--max_steps')+1]);lr=float(command[command.index('--learning_rate')+1])
        records(target,steps,10.,lr)
        return 30.
    monkeypatch.setattr(module,'run',run)
    monkeypatch.setenv('KERNEL_ENV_URL','http://127.0.0.1:8000')
    out=root/'pilot/new'
    monkeypatch.setattr(sys,'argv',['better_campaign.py','--root',str(root),'--phase','pilot',
        '--out',str(out),'--deadline','9999999999','--reuse-control',str(control)])
    module.main()
    result=json.loads((out/'result.json').read_text())
    assert result['passed'] and result['pipeline']==4
    assert len(calls)==2  # Only the missing pipelined control and RL sanity run execute.
    assert result['measurements']['0']['wall_seconds'] is None
    assert json.loads((root/'selection.json').read_text())['pipeline']==4


def test_pilot_aggregates_all_updates_for_rollout_cost(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir()
    metrics=[{'step':i,'rollout/wall_seconds':20.,'rollout/generation_seconds':3.,
       'rollout/generated_tokens':30.,'rollout/tokens_per_generation_second':10.,
       'step_time':25.,'loss':.01,'grad_norm':.02} for i in (1,2)]
    (adapter/'live_trainer_state.json').write_text(json.dumps({'global_step':2,'log_history':metrics}))
    (adapter/'rollouts-rank0.jsonl').write_text(''.join(json.dumps({'task_id':'relu',
       'turns':[{'stage':'timed','correct':True}]})+'\n' for i in range(2)))
    result=pilot_summary(tmp_path,60.)
    assert result['optimizer_updates']==2
    assert result['rollout_seconds']==40.
    assert result['generation_seconds']==6.
    assert result['generation_tokens_per_second']==10.
    assert result['completed_episodes_per_rollout_dollar']==pytest.approx(18.)


def test_partial_rl_requires_explicit_verified_new_run_checkpoint(tmp_path):
    from better_campaign import rl_artifact
    path=tmp_path/'experiment/seed42/adapter/checkpoint-24';path.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json'):(path/name).write_text('fixture')
    (path/'trainer_state.json').write_text(json.dumps({'global_step':24}))
    selection={'rl_checkpoint':str(path)}
    with pytest.raises(ValueError,match='provenance'):rl_artifact(tmp_path,selection)
    selection['rl_checkpoint_provenance']={'verified_intact':True,'path':str(path),'actual_step':24}
    assert rl_artifact(tmp_path,selection)==(path,24)
    selection['rl_checkpoint']=str(tmp_path/'old-v5/checkpoint-16')
    with pytest.raises(ValueError,match='new experiment'):rl_artifact(tmp_path,selection)


def test_partial_rl_rejects_mismatched_step_and_missing_weights(tmp_path):
    from better_campaign import rl_artifact
    path=tmp_path/'experiment/seed42/adapter/checkpoint-24';path.mkdir(parents=True)
    (path/'trainer_state.json').write_text(json.dumps({'global_step':24}))
    proof={'verified_intact':True,'path':str(path),'actual_step':23}
    selection={'rl_checkpoint':str(path),'rl_checkpoint_provenance':proof}
    with pytest.raises(ValueError,match='step provenance mismatch'):rl_artifact(tmp_path,selection)
    proof['actual_step']=24
    with pytest.raises(ValueError,match='files missing'):rl_artifact(tmp_path,selection)

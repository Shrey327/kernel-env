"""Two-seed, bounded GRPO campaign with a train-only curriculum probe and frozen evaluation."""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from baseline_eval import start_vllm_server, summarize
from kernel_env.server.tasks import load_tasks, build_prompt

BASE = 'Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
CANDIDATES = ['level1/'+name for name in (
    '19_ReLU','20_LeakyReLU','21_Sigmoid','22_Tanh','26_GELU_','27_SELU_',
    '29_Softplus','30_Softsign','31_ELU','1_Square_matrix_multiplication_',
    '3_Batched_matrix_multiplication','4_Matrix_vector_multiplication_',
    '6_Matmul_with_large_K_dimension_','7_Matmul_with_small_K_dimension_',
    '8_Matmul_with_irregular_shapes_','9_Tall_skinny_matrix_multiplication_',
    '12_Matmul_with_diagonal_matrices_','14_Matmul_for_upper_triangular_matrices',
    '15_Matmul_for_lower_triangular_matrices','16_Matmul_with_transposed_A',
    '17_Matmul_with_transposed_B','18_Matmul_with_transposed_both',
    '36_RMSNorm_','40_LayerNorm','47_Sum_reduction_over_a_dimension',
    '48_Mean_reduction_over_a_dimension','94_MSELoss','96_HuberLoss')]
RESERVED = ['level1/'+name for name in (
    '25_Swish','28_HardSigmoid','32_HardTanh','5_Matrix_scalar_multiplication',
    '10_3D_tensor_matrix_multiplication','11_4D_tensor_matrix_multiplication',
    '89_cumsum','91_cumsum_reverse')]
DIAGNOSTIC = CANDIDATES[:4]
EXCLUDED = {'level1/49_Max_reduction_over_a_dimension',
            'level1/53_Min_reduction_over_a_dimension', 'level1/100_HingeLoss'}


def benchmark_inference(model, tasks):
    from openai import OpenAI
    client=OpenAI(base_url='http://127.0.0.1:8001/v1',api_key='EMPTY',timeout=90,max_retries=0)
    prompts=[build_prompt(tasks[task],max_turns=2) for task in CANDIDATES[:8]]
    def generate(prompt):
        t=time.monotonic()
        result=client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],
            temperature=0,max_tokens=256,seed=1729)
        assert result.usage and result.usage.completion_tokens>0
        return {'tokens':result.usage.completion_tokens,'latency_s':time.monotonic()-t}
    try:
        generate(prompts[0])
        rows=[]
        for concurrency in (1,8):
            t=time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(generate,prompts))
            seconds=time.monotonic()-t
            rows.append({'concurrency':concurrency,'requests':len(results),'seconds':seconds,
                'generated_tokens':sum(r['tokens'] for r in results),
                'tokens_per_second':sum(r['tokens'] for r in results)/seconds,
                'request_latency_s':[r['latency_s'] for r in results]})
        return {'note':'Short generation-only throughput probe; prefix cache is warm and later tests benefit from reuse. Not a kernel quality comparison.',
                'max_tokens':256,'measurements':rows}
    finally:client.close()


def read_episodes(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def choose_curriculum(episodes, eligible):
    if any(ep.get('error') for ep in episodes):
        raise ValueError('probe contains infrastructure errors')
    solved = {ep['task_id'] for ep in episodes if any(t['correct'] for t in ep['turns'])}
    if not solved <= set(eligible):
        raise ValueError('probe task outside the frozen training candidates')
    return sorted(solved)


def compare_paired(before, after, groups):
    def keyed(episodes):
        rows = {}
        for ep in episodes:
            key = ep['task_id'], ep['sample_seed']
            if key in rows or ep.get('error') or not ep['turns']:
                raise ValueError('duplicate, incomplete or infrastructure-failed evaluation')
            rows[key] = ep
        return rows
    a, b = keyed(before), keyed(after)
    if set(a) != set(b):
        raise ValueError('before/after task and sample-seed sets differ')
    result = {}
    for name, ids in groups.items():
        selected = [key for key in a if key[0] in ids]
        if set(k[0] for k in selected) != set(ids):
            raise ValueError('missing task in '+name)
        diffs = []
        for task in ids:
            keys = [k for k in selected if k[0] == task]
            diffs.append(sum(int(any(t['correct'] for t in b[k]['turns'])) -
                             int(any(t['correct'] for t in a[k]['turns'])) for k in keys)/len(keys))
        rng = random.Random(1729)
        boot = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(4000))
        result[name] = {'before': summarize([a[k] for k in selected]),
                        'after': summarize([b[k] for k in selected]),
                        'paired_correctness_delta_pp': 100*sum(diffs)/len(diffs),
                        'task_bootstrap_95pct_delta_pp': [100*boot[100],100*boot[3899]],
                        'tasks': len(ids), 'episodes': len(selected)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--steps', type=int, default=64)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    tasks = load_tasks()
    heldout = sorted(t for t,v in tasks.items() if v.level==1 and v.split=='eval' and t not in EXCLUDED)
    assert set(CANDIDATES+RESERVED) <= {t for t,v in tasks.items() if v.split=='train'}
    assert not set(CANDIDATES)&set(RESERVED+heldout)
    groups = {'training_diagnostic':DIAGNOSTIC,'reserved_transfer':RESERVED,'official_heldout':heldout}
    protocol = {'model':BASE,'model_revision':REVISION,'steps':a.steps,'training_seeds':[42,123],
                'groups':groups,'candidate_tasks':CANDIDATES,'eval_samples':4,'eval_temperature':0.7,
                'eval_seed':1729,'max_turns':2,'max_new_tokens':2048,'episodes_per_update':32,
                'rollout_batch_size':8,'budget_cap_usd':100,'prior_reported_usd':7.24075504,
                'deadline_unix':a.deadline,'created_at_unix':time.time(),
                'verifier_version':'signed-activations-regression-reduction-v2',
                'hardware':'2 H100 in one container; model GPU0, evaluator GPU1; no region pin',
                'resource_rate_upper_usd_per_hour':9.7,
                'learning_reward_policy':'timing-equivalence ties v1: 3 warmed medians; max(2 percent, 1us, observed range); identical-source ties; scale_rewards none',
                'inference':'bf16; vLLM continuous batching, prefix cache, chunked prefill; no quantization'}
    import importlib.metadata
    provenance={'versions':{name:importlib.metadata.version(name) for name in
        ('torch','triton','trl','transformers','peft','vllm','openenv-core')},
        'model_revision':REVISION,'kernelbench_revision':'423217d9fda91e0c2d67e4a43bf62f96f6d104f1',
        'source_sha256':{str(f):__import__('hashlib').sha256(f.read_bytes()).hexdigest()
            for folder in (Path('/root/train'),Path('/root/kernel_env')) for f in folder.rglob('*.py')}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2))
    for seed in (42,123):
        folder=out/f'seed{seed}';folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps({**protocol,'seed':seed},indent=2))

    def status(phase, seed=None, **extra):
        data={'phase':phase,'active_seed':seed,'updated_unix':time.time(),**extra}
        tmp=out/'status.json.tmp';tmp.write_text(json.dumps(data));tmp.replace(out/'status.json')
        print('CAMPAIGN',json.dumps(data),flush=True)

    def run(command, path, timeout):
        if timeout <= 0:raise TimeoutError('no phase budget remains')
        started=time.monotonic()
        with path.open('w') as f:
            proc=subprocess.Popen([sys.executable,'-u',*command],stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True)
            try:rc=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError(str(path))
        print('PHASE_COMPLETE',json.dumps({'log':str(path),'seconds':time.monotonic()-started,'returncode':rc}),flush=True)
        if rc:raise RuntimeError(f'phase failed ({rc}): {path}')

    from huggingface_hub import snapshot_download
    base=snapshot_download(BASE,revision=REVISION)
    common=['baseline_eval.py','--env-url',os.environ['KERNEL_ENV_URL'],'--backend','openai',
            '--max-new-tokens','2048','--samples','4','--temperature','0.7','--seed','1729']
    eval_ids=DIAGNOSTIC+RESERVED+heldout

    def evaluations(model, rows):
        server=None
        try:
            server=start_vllm_server(model,8001,8192,1,0,
                '--enable-prefix-caching --enable-chunked-prefill --max-num-seqs 32 '
                '--max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --generation-config vllm')
            if rows[0][1]==out/'probe.jsonl':
                status('Inference throughput probe')
                (out/'inference_benchmark.json').write_text(json.dumps(benchmark_inference(model,tasks),indent=2))
                status('Baseline and training probe')
            for ids,path in rows:
                run([*common,'--model',model,'--task-ids',*ids,'--out',str(path)],path.with_suffix('.log'),
                    min(3600,a.deadline-time.time()-300))
        finally:
            if server:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()

    try:
        status('Baseline and training probe')
        evaluations(base,[(CANDIDATES,out/'probe.jsonl'),(eval_ids,out/'before.jsonl')])
        curriculum=choose_curriculum(read_episodes(out/'probe.jsonl'),CANDIDATES)
        if not curriculum:raise RuntimeError('no reward-bearing training task; no optimizer run launched')
        (out/'train_tasks.json').write_text(json.dumps(curriculum,indent=2))
        (out/'curriculum_selection.json').write_text(json.dumps({'selected':curriculum,'policy':'any correct probe episode; training candidates only'},indent=2))
        before=read_episodes(out/'before.jsonl')
        for seed in (42,123):
            folder=out/f'seed{seed}'
            status('Training',seed,selected_tasks=len(curriculum))
            remaining_seeds=2 if seed==42 else 1
            seconds=min(10800,(a.deadline-time.time()-3600)/remaining_seeds-900)
            train_stop=time.time()+seconds
            cmd=['grpo_kernel.py','--model_name_or_path',base,'--dtype','bfloat16','--bf16','true',
                 '--env_url',os.environ['KERNEL_ENV_URL'],'--task_file',str(out/'train_tasks.json'),
                 '--use_peft','--lora_r','16','--lora_alpha','32','--lora_target_modules','all-linear',
                 '--use_vllm','--vllm_mode','colocate','--vllm_gpu_memory_utilization','0.4',
                 '--vllm_max_model_length','8192','--num_generations','4',
                 '--per_device_train_batch_size','4','--gradient_accumulation_steps','8',
                 '--max_completion_length','2048','--max_steps',str(a.steps),'--learning_rate','2e-5',
                 '--gradient_checkpointing','true','--max_grad_norm','0.05','--loss_type','dr_grpo',
                 '--scale_rewards','none',
                 '--return_mode','discounted','--gamma','0.4','--temperature','0.8',
                 '--rollout_batch_size','8','--stop_at_unix',str(train_stop),
                 '--seed',str(seed),'--logging_steps','1','--report_to','none',
                 '--save_steps','4','--save_total_limit','2','--output_dir',str(folder/'adapter')]
            run(cmd,folder/'train.log',seconds+600)
            status('Merge',seed)
            run(['merge_lora.py','--base',base,'--adapter',str(folder/'adapter'),'--out',str(folder/'merged')],
                folder/'merge.log',min(600,a.deadline-time.time()-300))
            status('Final evaluation',seed)
            evaluations(str(folder/'merged'),[(eval_ids,folder/'after.jsonl')])
            comparison=compare_paired(before,read_episodes(folder/'after.jsonl'),groups)
            (folder/'comparison.json').write_text(json.dumps(comparison,indent=2))
            status('Seed complete',seed,comparison=comparison)
        comparisons={str(seed):json.loads((out/f'seed{seed}/comparison.json').read_text()) for seed in (42,123)}
        (out/'comparison.json').write_text(json.dumps(comparisons,indent=2))
        status('Complete',comparisons=comparisons)
    except BaseException as error:
        status('Stopped with incomplete work',error=repr(error))
        raise


if __name__=='__main__':main()

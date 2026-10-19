"""Identity, source and artifact-bound gates for the corrected experiment."""
import hashlib,json
from pathlib import Path

RUN='corrected-20261019'
BASE='Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
from kernel_env.server.cuda_graph_timing import METHOD


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_files(root):
    root=Path(root)
    paths=[*root.joinpath('kernel_env').rglob('*.py'),*root.joinpath('train').glob('*.py'),root/'modal_corrected.py',root/'launch_corrected.py',root/'modal_app.py']
    result={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            if p.exists() and not any(x in ('.venv','__pycache__','tests') for x in p.parts) and not p.name.startswith('test_')}
    for relative in ('kernel_env/tests/kernels.py','kernel_env/tests/fixtures/level1/19_ReLU.py'):
        result[relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    return result


def verify_sources(root,expected):
    for name,want in expected.items():
        if name in ('launch_corrected.py','modal_app.py'):continue  # host-only guard is separately hashed in the reservation
        path=Path(root)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=want:
            raise ValueError('source mismatch: '+name)


def require_gate(gate,contract,stage):
    if gate.get('passed') is not True or gate.get('stage')!=stage:raise ValueError('missing passed '+stage+' gate')
    for key in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256'):
        if not contract.get(key) or gate.get(key)!=contract[key]:raise ValueError(stage+' gate mismatch: '+key)


def protocol(sft):
    from better_protocol import CURRICULUM,variant_source,TRAINING_SHAPES,TRAINING_VARIANT
    from kernel_env.server.tasks import load_tasks
    tasks=load_tasks()
    from campaign import DIAGNOSTIC,RESERVED,EXCLUDED
    return {'run_id':RUN,'model':BASE,'revision':REVISION,'timing_method':METHOD,
       'measurement':'Amortized CUDA graph replay event interval, with unsubtracted launch/submission/idle overhead; host guard/allocation/validation/profiling outside event calls; captured GPU copies included. Not pure kernel time.',
       'warmup':'1validated eager call per arm before capture;3validated graph replays per arm before first timed sample;3null graph warmups/canary checks before7null samples; no measured samples discarded; evidence retained separately; synthetic1.1x sensitivity checked for every fixed replicate',
       'graph_policy':'Fixed512MiB independent input-storage target; repeats=max(1,min(64,floor(512MiB/input_bytes))); full live-memory admission cap64GiB with8GiB device reserve and256MiB comparison scratch allowance; fail rather than adapt repeat count; every timed replay checked; no alternate timer',
       'uncertainty':'max(2%median, observed range, calibrated null launch and0.5us/event divided by repeat count); no subtraction; engineering bounds',
       'curriculum':CURRICULUM,'training_shapes':TRAINING_SHAPES,'training_variant':TRAINING_VARIANT,
       'shape_rationale':'Training activations/reductions32MiB and losses32MiB total inputs;512MiB graph input storage amortizes launch uncertainty and exceeds H100 cache capacity.32/64MiB variants reverified, matmul1024/1536. Shapes fixed before empirical measurements; no selection based on favorable results.',
       'sft_shape_provenance':'Reuse exact clean12-step SFT artifact trained on prior small shapes; no new SFT updates. Saved SFT dataset hash remains unchanged. Reverify all28 handwritten teachers separately on the new fixed32/64MiB variants before pilot.',
       'sensitivity':'All3 fixed identity-control replicates at each required new32/64MiB regime must resolve synthetic1.1x. Old2MiB control is retained as a diagnostic of the known insensitive regime, excluded from training. Synthetic ratios are not achieved speedups.',
       'validation_policy':'Predeclared required cases; collect independent outcomes after ordinary failure while GPU healthy and deadline allows; retain all samples and failures; any required failed/not-run case fails overall gate. No automatic retry or outlier rejection.',
       'stream_policy':'Candidate eager and captured replay profiles must show exactly one CUDA device/stream; missing stream evidence or additional streams are rejected.',
       'validation_seconds':1800,'original_shape_eval_timeout_seconds':600,'validation_no_progress_seconds':660,
       'evaluation_shapes':'original pinned KernelBench shapes; Softsign4096x393216 float32=6GiB',
       'evaluation_groups':{'training_diagnostic':DIAGNOSTIC,'reserved_transfer':RESERVED,'official_heldout':sorted(k for k,t in tasks.items() if t.level==1 and t.split=='eval' and k not in EXCLUDED)},
       'task_sources_sha256':{tid:hashlib.sha256(task.src.encode()).hexdigest() for tid,task in tasks.items()},
       'training_variant_sha256':{tid+'/'+str(v):hashlib.sha256(variant_source(tasks[tid],v).encode()).hexdigest() for tid in CURRICULUM for v in (0,1)},
       'excluded':sorted(EXCLUDED),'evaluation_samples':1,'evaluation_temperature':.7,'evaluation_seed':1729,
       'training_steps':32,'episodes_per_update':8,'generations':4,'attempts':3,'gamma':1.0,
       'credit':'independent trajectory returns; N/(N-1) trajectory LOO; synchronous optimizer barrier; not turn-levelTRLOO',
       'reward':'corrected-speed-progress-v2: invalid0; correct0.25+clip(log2(conservative speedup),0,2); best validated quality within3attempts; telescoping progress credits; gamma1',
       'training_temperature':.8,'max_new_tokens':1536,'context_length':12288,'sampling':'vLLM colocated,35percent memory, synchronous; no overlap sampler',
       'multi_objective_aggregation':'sum_then_normalize','scale_rewards':'none','kl_beta':0.,
       'learning_rate':1.5e-5,'lora_r':16,'dtype':'bfloat16','training_start':'clean12-step verified SFT; never biased RL checkpoint',
       'sft_sha256':sft['adapter_sha256'],'teacher_dataset_sha256':sft['examples_sha256'],'fresh_cap_usd':100,
       'kernelbench_commit':'423217d9fda91e0c2d67e4a43bf62f96f6d104f1',
       'authorization_boundary_utc':'2026-10-19T16:01:11Z','gpu':'H100:2, no region pin, no nonpreemptible flag'}

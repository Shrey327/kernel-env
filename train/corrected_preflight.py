"""Bounded GPU preflight; ordinary failures retain every independent diagnostic."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from importlib.metadata import version
from corrected_contract import METHOD,verify_sources


class BoundedCases:
    """An upfront manifest, durable outcomes, and a fail-closed aggregate."""
    def __init__(self,out,manifest,deadline,progress_path,clock=time.time):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.manifest=manifest;self.deadline=deadline;self.clock=clock;self.progress_path=Path(progress_path)
        self.results={};self.unhealthy=None;self.health=None
        if len({r['case'] for r in manifest})!=len(manifest):raise ValueError('duplicate manifest case')
        self.write('case-manifest.json',{'deadline_unix':deadline,'cases':manifest})

    def write(self,name,value):
        target=self.out/name;temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2));temporary.replace(target)

    def remaining(self):return self.deadline-self.clock()-20

    def progress(self,label):
        target=self.progress_path;temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps({'semantic_progress':label,'updated_unix':self.clock()}));temporary.replace(target)

    def record(self,label,result,passed,status=None,reason=None):
        spec=next(row for row in self.manifest if row['case']==label)
        row={'case':label,'required':spec.get('required',True),'status':status or ('passed' if passed else 'failed'),
             'passed':bool(passed),'result':result,'finished_unix':self.clock()}
        if reason:row['reason']=reason
        self.results[label]=row
        with (self.out/'cases.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps({key:row[key] for key in ('case','passed','status')}),flush=True)
        return row

    def run(self,label,operation,requires=(),gpu=False):
        if label in self.results:raise ValueError('case already completed: '+label)
        blocked=[name for name in requires if not self.results.get(name,{}).get('passed')]
        reason=('dependency failed or not run: '+', '.join(blocked)) if blocked else None
        if self.remaining()<=0:reason='reserved preflight deadline'
        if gpu and self.unhealthy:reason='GPU unhealthy: '+self.unhealthy
        if reason:return self.record(label,{},False,'notrun',reason)
        self.progress(label);started=self.clock()
        try:
            if gpu and self.health:self.health()
            result,passed=operation()
            return self.record(label,{**result,'case_wall_seconds':self.clock()-started},passed)
        except Exception as error:
            result={'error':type(error).__name__+': '+str(error),'case_wall_seconds':self.clock()-started}
            if gpu and self.health:
                try:self.health()
                except Exception as health_error:self.unhealthy=type(health_error).__name__+': '+str(health_error)
            return self.record(label,result,False)

    def finish(self,contract,**extra):
        for spec in self.manifest:
            if spec['case'] not in self.results:self.record(spec['case'],{},False,'notrun','case was not reached')
        failures=[row['case'] for row in self.results.values() if row['required'] and not row['passed']]
        gate={key:contract[key] for key in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')}
        gate.update(stage='gpu-validation',passed=not failures,cases=len(self.results),failures=failures,
                    not_run=[row['case'] for row in self.results.values() if row['status']=='notrun'],
                    outcomes=list(self.results.values()),finished_unix=self.clock(),**extra)
        self.write('result.json',gate);self.progress('Validation complete' if gate['passed'] else 'Validation complete; launch gate failed')
        return gate


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--contract',required=True);p.add_argument('--progress',required=True);p.add_argument('--deadline',required=True,type=float)
    a=p.parse_args(argv);out=Path(a.out);contract=json.loads(Path(a.contract).read_text())
    from better_protocol import CURRICULUM,teacher_source,variant_source
    identity_shapes=((32,16384),(32,262144),(64,262144))
    identity_names=[str(shape) for shape in identity_shapes];required_regimes=identity_names[1:]
    teacher_names=['teacher '+tid+'/'+str(variant) for tid in CURRICULUM for variant in (0,1)]
    adversary_names=['unjoined stream short','unjoined stream long','late zeros','cached output','late mutation','graph incompatible']
    manifest=[{'case':name,'required':True} for name in ('source identity','pinned integration','task source identity','clean SFT identity')]
    manifest += [{'case':'identity '+name,'required':name in required_regimes} for name in identity_names]
    manifest += [{'case':name,'required':True} for name in ['installed trainer credit','reward calibration','known fused original6GiB Softsign',*adversary_names,
                 *['fast-kernel noise replicate '+str(i+1) for i in range(3)],'fast-kernel residual reward noise',*teacher_names]]
    cases=BoundedCases(out,manifest,a.deadline,a.progress);runtime={};controls={};reward={};repeats=[]
    trusted=('source identity','pinned integration','task source identity','clean SFT identity')
    try:
        cases.run('source identity',lambda:(verify_sources(Path(__file__).resolve().parent.parent,contract['source_files']) or {'verified_files':len(contract['source_files'])},True))
        def integration():
            nonlocal runtime
            runtime={name:version(name) for name in ('torch','triton','trl','vllm','peft','transformers','accelerate','datasets')}
            assert runtime['trl']=='1.16.0' and runtime['torch'].split('+')[0]=='2.13.0' and runtime['triton']=='3.7.1',runtime
            from trl import GRPOTrainer
            from inspect import signature
            assert 'rollout_func' in signature(GRPOTrainer.__init__).parameters
            return runtime,True
        cases.run('pinned integration',integration,requires=('source identity',))
        os.environ['KERNELBENCH_DIR']='/opt/KernelBench/KernelBench'
        from kernel_env.server.tasks import load_tasks
        tasks={}
        def task_identity():
            tasks.update(load_tasks())
            for tid,want in contract['protocol']['task_sources_sha256'].items():
                assert hashlib.sha256(tasks[tid].src.encode()).hexdigest()==want,('task source mismatch',tid)
            for tid in CURRICULUM:
                for variant in (0,1):
                    assert hashlib.sha256(variant_source(tasks[tid],variant).encode()).hexdigest()==contract['protocol']['training_variant_sha256'][tid+'/'+str(variant)]
            return {'task_count':len(tasks),'training_variants':len(CURRICULUM)*2},True
        cases.run('task source identity',task_identity,requires=('source identity','pinned integration'))
        def sft_identity():
            folder=Path('/runs')/contract['sft_source'];expected=json.loads((folder/'result.json').read_text())
            assert expected['completed'] and expected['steps']==12 and expected['examples']==28
            assert hashlib.sha256((folder/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==contract['sft_sha256']
            dataset=folder/'examples.jsonl'
            assert hashlib.sha256(dataset.read_bytes()).hexdigest()==contract['teacher_dataset_sha256']
            assert len(dataset.read_text().splitlines())==28
            return {'completed_steps':12,'examples':28,'adapter_sha256':contract['sft_sha256'],'dataset_sha256':contract['teacher_dataset_sha256'],
                    'scope':'Exact existing SFT weights and old training prompts; the following 28 teacher checks use the new fixed RL shapes, not a claim that SFT trained on those shapes.'},True
        cases.run('clean SFT identity',sft_identity,requires=('source identity','pinned integration'))
        if all(cases.results[name]['passed'] for name in trusted):
            import torch
            from kernel_env.server.worker import kernel_only_mode
            from kernel_env.server.validation_memory import input_fingerprint
            from kernel_env.server.cuda_graph_timing import capture_work,input_batches,checked_measure,checked_eager_warmup,calibrated_floor,graph_memory_plan,graph_repeats,cuda_memory_evidence
            from kernel_env.server.benchmark_protocol import paired_timings
            from kernel_env.server.evaluator import Evaluator
            def health():
                torch.empty(1,device='cuda').add_(1);torch.cuda.synchronize()
            cases.health=health
            def identity(shape):
                task=tasks['level1/30_Softsign'];file=out/'identity_control.py';file.write_text(teacher_source(task))
                spec=importlib.util.spec_from_file_location('corrected_control',file);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
                model=module.ModelNew().cuda().eval();torch.cuda.reset_peak_memory_stats()
                input_bytes=4*shape[0]*shape[1]
                memory_plan=graph_memory_plan(input_bytes,input_bytes,graph_repeats(input_bytes))
                x=torch.randn(shape,device='cuda');inputs=[x];batches=input_batches(torch,inputs)
                def guarded(*xs):
                    with kernel_only_mode():return model(*xs)
                with torch.no_grad():
                    fingerprint=lambda values:input_fingerprint(torch,values)
                    error=lambda r,n:None if torch.allclose(r,n,atol=.001,rtol=.001) else 'wrong values'
                    target=lambda v:v/(1+v.abs())
                    warmup=lambda fn,xs:checked_eager_warmup(torch,fn,xs,target,fingerprint,error)
                    reference=capture_work(torch,model,inputs,'cuda',batches,warmup)
                    candidate=capture_work(torch,guarded,inputs,'cuda',batches,warmup)
                    floor,calibration=calibrated_floor(torch,candidate,'cuda');counter=[0]
                    def measure(work):
                        if cases.remaining()<=0:raise TimeoutError('reserved preflight deadline during fixed identity samples')
                        pair=counter[0]//2;counter[0]+=1
                        generator=torch.Generator(device='cuda').manual_seed(1729+pair)
                        fresh=[[torch.randn(shape,device='cuda',generator=generator)] for _ in batches]
                        return checked_measure(torch,work,fresh,target,fingerprint,error,'cuda')
                    replicates=[]
                    for _ in range(3):
                        replicates.append(paired_timings(lambda:measure(reference),lambda:measure(candidate),seed=42,absolute_floor_ms=floor))
                    for replicate in replicates:replicate['timing_method']=METHOD
                result={**replicates[0],'warmup_evidence':{'reference_eager':reference.eager_warmup_evidence,'candidate_eager':candidate.eager_warmup_evidence,
                        'reference_graph':reference.warmup_evidence,'candidate_graph':candidate.warmup_evidence},'calibration':calibration,'replicates':replicates,
                        'input_shape':list(shape),'input_bytes':x.numel()*x.element_size(),'memory_plan':memory_plan,
                        'memory_evidence':cuda_memory_evidence(torch,'cuda')}
                controls[str(shape)]=result
                return result,all(.90<=r['speedup']<=1.10 and r['speedup_lower_bound']<=1<=r['speedup_upper_bound'] for r in replicates)
            for shape in identity_shapes:
                cases.run('identity '+str(shape),lambda shape=shape:identity(shape),requires=trusted,gpu=True)
                if not cases.unhealthy:
                    try:torch.cuda.empty_cache()
                    except Exception as error:cases.unhealthy=type(error).__name__+': '+str(error)
            trainer={}
            def trainer_credit():
                from verify_trainer_credit import verify
                trainer.update(verify());return trainer,trainer.get('passed') is True
            cases.run('installed trainer credit',trainer_credit,requires=trusted)
            def calibration():
                from reward_calibration_artifact import calibrate
                reward.update(calibrate(controls,trainer,required_regimes=required_regimes));cases.write('reward-calibration.json',reward)
                return reward,reward['passed']
            cases.run('reward calibration',calibration,requires=trusted)
            ev=Evaluator(gpus=['0'],workers_per_gpu=1,timeout_s=45,warmup_ms=3,rep_ms=10)
            def evaluate(ref,code,tid,seeds=(191,192,193),timeout=45):
                ev.timeout_s=min(timeout,max(1,cases.remaining()))
                return ev.evaluate(ref,code,'triton',list(seeds),task_id=tid)
            soft=tasks['level1/30_Softsign']
            def fused():
                value=evaluate(soft.src,teacher_source(soft),soft.task_id,timeout=600)
                return value,value.get('correct') and value.get('timing_method')==METHOD and value.get('speedup_lower_bound',0)>1.25
            cases.run('known fused original6GiB Softsign',fused,requires=trusted,gpu=True)
            from kernel_env.tests.kernels import GOOD
            fixture=Path('/root/kernel_env/tests/fixtures/level1/19_ReLU.py').read_text()
            late=GOOD.replace('super().__init__()','super().__init__(); self.calls=0').replace('n = x.numel()','self.calls += 1\n        n = x.numel()').replace('return y','return torch.zeros_like(y) if self.calls>8 else y')
            cached=GOOD.replace('super().__init__()','super().__init__(); self.cached=None').replace('return y','if self.cached is None:\n            self.cached=y\n        return self.cached')
            mutation='\n@triton.jit\ndef mutate_kernel(X,N:tl.constexpr,B:tl.constexpr):\n    i=tl.program_id(0)*B+tl.arange(0,B)\n    tl.store(X+i,0.,i<N)\n'
            mutated=GOOD.replace('class ModelNew',mutation+'\nclass ModelNew').replace('super().__init__()','super().__init__(); self.calls=0').replace('n = x.numel()','self.calls += 1\n        n = x.numel()').replace('return y','if self.calls>8:\n            mutate_kernel[(triton.cdiv(n,1024),)](x,n,1024)\n        return y')
            incompatible=GOOD.replace('return y','torch.cuda.synchronize()\n        return y')
            unjoined=GOOD.replace('super().__init__()','super().__init__(); self.side=torch.cuda.Stream()').replace('return y','with torch.cuda.stream(self.side):\n            torch.cuda._sleep(100000)\n        return y')
            unjoined_long=unjoined.replace('100000','20000000')
            from kernel_env.server.calibrated_reward import quality
            for name,code in zip(adversary_names,(unjoined,unjoined_long,late,cached,mutated,incompatible)):
                def adversary(name=name,code=code):
                    value=evaluate(fixture,code,'level1/19_ReLU',seeds=(1,2,3))
                    reject=value.get('stage') in ('format','correctness') and not value.get('correct')
                    if name=='graph incompatible':reject=reject and 'no alternate timer' in value.get('feedback','')
                    value['policy_reward']=quality(bool(value.get('correct')),value.get('speedup_lower_bound'),value.get('speedup_upper_bound'))
                    return value,reject and value['policy_reward']==0
                cases.run(name,adversary,requires=trusted,gpu=True)
            for index in range(3):
                def noise():
                    from better_protocol import TRAINING_VARIANT
                    value=evaluate(variant_source(soft,TRAINING_VARIANT),teacher_source(soft),soft.task_id)
                    passed=value.get('correct') and value.get('timing_method')==METHOD
                    if passed:value['policy_reward']=quality(True,value['speedup_lower_bound'],value['speedup_upper_bound']);repeats.append(value)
                    return value,passed
                cases.run('fast-kernel noise replicate '+str(index+1),noise,requires=trusted,gpu=True)
            def residual_noise():
                import statistics
                margins=[row['per_turn_quality']-.25 for regime in reward.get('regimes',{}).values() if regime['required_for_training']
                         for row in regime.get('cases',[]) if row['case']=='reliable1.1x']
                complete=len(repeats)==3 and len(margins)==len(required_regimes)*3
                spread=statistics.pstdev(row['policy_reward'] for row in repeats) if len(repeats)==3 else None
                margin=min(margins) if margins else None
                value={'replicates':repeats,'reward_std':spread,'reliable1_1x_reward_margin':margin,'complete':complete,
                       'noise_gate':'reliable1.1x bonus must exceed3x observed fast-code reward std; engineering check, not statistical confidence',
                       'passed':complete and margin>3*spread}
                reward['synthetic_calibration_passed']=reward.get('passed',False)
                reward['identical_fast_source_against_pytorch']=value
                reward['passed']=reward['synthetic_calibration_passed'] and value['passed']
                cases.write('reward-calibration.json',reward)
                return value,value['passed']
            cases.run('fast-kernel residual reward noise',residual_noise,requires=trusted)
            for tid in CURRICULUM:
                for variant in (0,1):
                    def teacher(tid=tid,variant=variant):
                        value=evaluate(variant_source(tasks[tid],variant),teacher_source(tasks[tid]),tid,timeout=30)
                        value.update(task_id=tid,variant=variant,code_sha256=hashlib.sha256(teacher_source(tasks[tid]).encode()).hexdigest())
                        return value,value.get('correct') and value.get('timing_method')==METHOD
                    cases.run('teacher '+tid+'/'+str(variant),teacher,requires=trusted,gpu=True)
        else:
            for spec in manifest:
                if spec['case'] not in cases.results:cases.run(spec['case'],lambda:({},False),requires=trusted)
    except Exception as error:
        cases.write('failure-report.json',{'error':type(error).__name__+': '+str(error),'finished_unix':time.time()})
    finally:
        gate=cases.finish(contract,runtime_versions=runtime)
        if reward and 'identical_fast_source_against_pytorch' not in reward:
            reward['synthetic_calibration_passed']=reward.get('passed',False)
            reward['passed']=False
            reward['identical_fast_source_against_pytorch']={'passed':False,'complete':False,'replicates':repeats,
                'reward_std':None,'reliable1_1x_reward_margin':None,'reason':'residual noise aggregate failed or was not run; see cases.jsonl'}
            cases.write('reward-calibration.json',reward)
        # Stage gates exist only when their full required evidence passed.
        binding={key:gate[key] for key in ('run_id','source_digest','protocol_digest','sft_sha256','teacher_dataset_sha256')}
        stage_cases={'gpu-validation':[r['case'] for r in manifest if r.get('required',True)],
                     'reward-calibration':['reward calibration','fast-kernel residual reward noise','installed trainer credit',
                                           *['identity '+name for name in required_regimes],*adversary_names],
                     'sft-verification':['clean SFT identity',*teacher_names]}
        gates={}
        for stage,names in stage_cases.items():
            if all(cases.results[name]['passed'] for name in names):
                gates[stage]={**binding,'stage':stage,'passed':True,'cases':names,'finished_unix':time.time()}
                if stage=='reward-calibration':
                    noise=reward['identical_fast_source_against_pytorch']
                    gates[stage].update(fast_code_reward_std=noise['reward_std'],minimum_1_1x_reward_margin=noise['reliable1_1x_reward_margin'])
        cases.write('gates.json',gates)
    return gate


if __name__=='__main__':main()

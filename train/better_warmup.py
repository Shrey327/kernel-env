"""Short assistant-only SFT on handwritten kernels verified on training tasks only."""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

from better_protocol import CURRICULUM, variant_source, teacher_source
from kernel_env.server.tasks import load_tasks, build_prompt

BASE='Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'


def encode_example(tokenizer,messages):
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    if not text.startswith(prefix):raise ValueError('assistant text prefix mismatch')
    row=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if len(row['input_ids'])>4096:raise ValueError('verified example exceeds context budget')
    # BPE may merge the last prefix newline with the code fence; never supervise that mixed token.
    offsets=row.pop('offset_mapping')
    row['labels']=[token if start>=len(prefix) and end>start else -100
                   for token,(start,end) in zip(row['input_ids'],offsets)]
    if all(label==-100 for label in row['labels']):raise ValueError('no assistant tokens')
    return row


def verified_examples(tasks):
    return [{'task_id':tid,'variant':variant,'messages':[
        {'role':'user','content':build_prompt(dataclasses.replace(tasks[tid],src=variant_source(tasks[tid],variant)),max_turns=3)},
        {'role':'assistant','content':'```python\n'+teacher_source(tasks[tid])+'\n```'}]}
        for tid in CURRICULUM for variant in (0,1)]


def reuse_verified(folder,expected):
    try:
        examples=[json.loads(line) for line in (folder/'examples.jsonl').read_text().splitlines()]
        rows=[json.loads(line) for line in (folder/'verified.jsonl').read_text().splitlines()]
    except FileNotFoundError:return None
    if examples!=expected or len(rows)!=len(expected):return None
    for row,example in zip(rows,expected):
        code=example['messages'][-1]['content'][10:-4]
        if (row.get('task_id'),row.get('variant'))!=(example['task_id'],example['variant']):return None
        if not row.get('correct') or row.get('stage')!='timed':return None
        if row.get('code_sha256')!=hashlib.sha256(code.encode()).hexdigest():return None
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    def progress(stage,**extra):
        path=root/'progress.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps({'stage':stage,'updated_unix':time.time(),**extra}));temp.replace(path)
    progress('Checking verified training examples')
    from kernel_env.server.evaluator import Evaluator
    tasks=load_tasks();ev=Evaluator(gpus=['0'],workers_per_gpu=1,timeout_s=90,warmup_ms=3,rep_ms=10)
    examples=verified_examples(tasks)
    prior=None;reused_from=None
    for path in sorted(root.parent.glob('*/examples.jsonl'),reverse=True):
        prior=reuse_verified(path.parent,examples)
        if prior is not None:reused_from=str(path.parent);break
    with (root/'verified.jsonl').open('w') as log:
        for index,example in enumerate(examples):
            tid,variant=example['task_id'],example['variant']
            task=tasks[tid];assert task.split=='train'
            progress('Verifying teacher kernels',verified=index,total=len(examples),task_id=tid)
            code=teacher_source(task)
            if prior is not None:
                result=prior[index]
            else:
                if time.time()>a.deadline-420:raise TimeoutError('verification used SFT reserve')
                shaped=dataclasses.replace(task,src=variant_source(task,variant))
                result=ev.evaluate(shaped.src,code,'triton',[191,192,193],task_id=tid)
            log.write(json.dumps({'task_id':tid,'variant':variant,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),**result})+'\n');log.flush()
            if result.get('stage')!='timed' or not result.get('correct'):
                raise RuntimeError(f'verified warmup gate failed: {tid}/{variant}: {result}')
    (root/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    import torch
    from datasets import Dataset
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,DataCollatorForSeq2Seq
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    dataset=Dataset.from_list([encode_example(tokenizer,e['messages']) for e in examples])
    progress('Loading SFT base model',verified=len(examples),reused_from=reused_from)
    model=AutoModelForCausalLM.from_pretrained(BASE,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'})
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules='all-linear',task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.config.use_cache=False
    class Stop(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            progress('SFT training',step=state.global_step,target=12,metrics=logs or {})
        def on_step_end(self,args,state,control,**kwargs):
            if time.time()>=a.deadline-60:control.should_training_stop=True
    args=TrainingArguments(output_dir=str(root/'adapter'),per_device_train_batch_size=1,
        gradient_accumulation_steps=2,max_steps=12,learning_rate=1e-4,bf16=True,
        gradient_checkpointing=True,logging_steps=1,report_to='none',save_strategy='no',seed=42)
    trainer=Trainer(model=model,args=args,train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100),callbacks=[Stop()])
    started=time.monotonic();metrics=trainer.train().metrics
    if trainer.state.global_step!=12:raise RuntimeError('SFT incomplete; do not advance pilot')
    trainer.save_model(str(root/'adapter'));tokenizer.save_pretrained(root/'adapter')
    trainer.state.save_to_json(root/'trainer_state.json')
    (root/'result.json').write_text(json.dumps({'completed':True,'examples':len(examples),'steps':12,
       'seconds':time.monotonic()-started,'metrics':metrics,'finished_unix':time.time(),
       'training_only':True,'teacher':'handwritten, GPU verified; no external teacher API',
       'base':BASE,'revision':REVISION,'reused_verified_examples_from':reused_from},indent=2))


if __name__=='__main__':main()

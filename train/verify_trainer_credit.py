"""Exercise installed TRL centering + real override on a tiny local model; no downloads/vLLM/GPU needed."""
import tempfile


def verify():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
    from datasets import Dataset
    from trl import GRPOConfig
    from grpo_kernel import TrajectoryLOOTrainer,env_return
    from kernel_env.server.calibrated_reward import loo_advantages
    vocab={'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'prompt':4,'answer':5}
    raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    model=GPT2LMHeadModel(GPT2Config(vocab_size=6,n_positions=16,n_embd=8,n_layer=1,n_head=1,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    values=[0.,1.,2.,4.]
    def rollout(prompts,trainer):
        return {'prompt_ids':[[4] for _ in prompts],'completion_ids':[[5,2] for _ in prompts],
                'logprobs':None,'episode_return':list(values)}
    with tempfile.TemporaryDirectory() as tmp:
        config=GRPOConfig(output_dir=tmp,use_cpu=True,bf16=False,fp16=False,per_device_train_batch_size=4,
            gradient_accumulation_steps=1,num_generations=4,max_completion_length=2,max_steps=1,
            scale_rewards='none',multi_objective_aggregation='sum_then_normalize',report_to='none',use_vllm=False,beta=0.)
        trainer=TrajectoryLOOTrainer(model=model,args=config,reward_funcs=env_return,processing_class=tokenizer,
             train_dataset=Dataset.from_dict({'prompt':['prompt']*4}),rollout_func=rollout)
        model.train();results=[]
        for group in ([0.,1.,2.,4.],[2.,2.,2.,2.],[2.,2.,2.,2.004],[10.,11.,12.,14.]):
            values[:]=group
            result=trainer._generate_and_score_completions([{'prompt':'prompt'} for _ in range(4)])
            got=result['advantages'].detach().cpu().tolist();want=loo_advantages(group)
            assert torch.allclose(torch.tensor(got),torch.tensor(want),atol=1e-6,rtol=1e-5),(group,got,want)
            assert all(__import__('math').isfinite(x) for x in got)
            if len(set(group))==1:assert got==[0.]*4
            results.append({'returns':group,'actual_trl_loo_advantages':got,'expected':want})
        return {'passed':True,'trl_version':__import__('importlib.metadata',fromlist=['version']).version('trl'),
                'base_trainer_method_sha256':__import__('hashlib').sha256(__import__('inspect').getsource(__import__('trl').GRPOTrainer._generate_and_score_completions).encode()).hexdigest(),
                'loo_override_sha256':__import__('hashlib').sha256(__import__('inspect').getsource(TrajectoryLOOTrainer._generate_and_score_completions).encode()).hexdigest(),
                'actual_method':'TrajectoryLOOTrainer._generate_and_score_completions with installed TRL, tiny CPU model and deterministic rollout',
                'scale_rewards':'none','multi_objective_aggregation':'sum_then_normalize','cases':results}


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))

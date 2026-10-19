import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from better_protocol import CURRICULUM,variant_source,teacher_source,training_copy,TRAINING_VARIANT
from better_warmup import encode_example,reuse_verified
from kernel_env.server.tasks import Task,load_tasks
from kernel_env.server.static_checks import check_source


def test_all_teachers_are_train_only_valid_source_with_both_shape_variants():
    import os
    if not os.environ.get('KERNELBENCH_DIR'):pytest.skip('pinned KernelBench source required')
    tasks=load_tasks()
    for tid in CURRICULUM:
        t=tasks[tid];assert t.split=='train'
        code=teacher_source(t);ast.parse(code)
        assert not check_source(code,'triton'), (tid,check_source(code,'triton'))
        for variant in (0,1):ast.parse(variant_source(t,variant))
    with pytest.raises(ValueError):variant_source(tasks['level1/23_Softmax'],0)


def test_assistant_only_labels_do_not_train_on_problem_or_feedback():
    class Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            return 'prompt\n' if add_generation_prompt else 'prompt\n```code'
        def __call__(self,text,add_special_tokens,return_offsets_mapping):
            return {'input_ids':[1,2,3,4,5],'attention_mask':[1]*5,
                    'offset_mapping':[(0,3),(3,6),(6,10),(10,12),(12,14)]}
    row=encode_example(Tokenizer(),[{'role':'user','content':'problem'},{'role':'assistant','content':'kernel'}])
    assert row['labels']==[-100,-100,-100,4,5]


def test_fixed_new_shapes_and_byte_counts_without_allocating_gpu_memory():
    import os
    import torch
    if not os.environ.get('KERNELBENCH_DIR'):pytest.skip('pinned KernelBench source required')
    tasks=load_tasks()
    assert TRAINING_VARIANT==0
    for tid in CURRICULUM:
        for variant in (0,1):
            ns={};exec(variant_source(tasks[tid],variant),ns)
            with torch.device('meta'):
                inputs=ns['get_inputs']();model=ns['Model'](*ns['get_init_inputs']())
                output=model(*inputs)
            size=sum(x.numel()*x.element_size() for x in inputs if isinstance(x,torch.Tensor))
            if tid.endswith('1_Square_matrix_multiplication_'):
                assert inputs[0].shape==(1024,1024) if variant==0 else inputs[0].shape==(1536,1536)
                assert size==[8,18][variant]*1024**2
            else:
                assert size==[32,64][variant]*1024**2, (tid,variant,size)
            assert output.numel()>0
            assert all(x.device.type=='meta' for x in inputs if isinstance(x,torch.Tensor))


def test_training_copy_keeps_heldout_source_original_and_uses_explicit_variant(tmp_path):
    import os
    if not os.environ.get('KERNELBENCH_DIR'):pytest.skip('pinned KernelBench source required')
    tasks=load_tasks();destination=training_copy(tasks,tmp_path/'tasks',variant=TRAINING_VARIANT)
    for tid,task in tasks.items():
        expected=variant_source(task,TRAINING_VARIANT) if tid in CURRICULUM else task.src
        assert (destination/(tid+'.py')).read_text()==expected


def test_pinned_tokenizer_masks_only_last_assistant_when_available():
    import os
    from transformers import AutoTokenizer
    path=os.environ.get('PINNED_TOKENIZER')
    if not path:pytest.skip('set PINNED_TOKENIZER to the pinned local tokenizer files')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    messages=[{'role':'user','content':'problem and feedback'},
              {'role':'assistant','content':'```python\nprint(1)\n```'}]
    row=encode_example(tokenizer,messages)
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    assert any(label!=-100 for label in row['labels'])
    assert all(label==-100 for label,(start,end) in zip(row['labels'],offsets) if start<len(prefix))


def test_reuse_requires_complete_exact_source_and_verified_code(tmp_path):
    import json,hashlib
    example={'task_id':'level1/19_ReLU','variant':0,'messages':[
        {'role':'user','content':'exact reference'},
        {'role':'assistant','content':'```python\ncode\n```'}]}
    expected=[example]
    (tmp_path/'examples.jsonl').write_text(json.dumps(example)+'\n')
    row={'task_id':example['task_id'],'variant':0,'correct':True,'stage':'timed',
         'code_sha256':hashlib.sha256(b'code').hexdigest()}
    (tmp_path/'verified.jsonl').write_text(json.dumps(row)+'\n')
    assert reuse_verified(tmp_path,expected)==[row]
    row['code_sha256']='tampered'
    (tmp_path/'verified.jsonl').write_text(json.dumps(row)+'\n')
    assert reuse_verified(tmp_path,expected) is None
    (tmp_path/'verified.jsonl').write_text('')
    assert reuse_verified(tmp_path,expected) is None

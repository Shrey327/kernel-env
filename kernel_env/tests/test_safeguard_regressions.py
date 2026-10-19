import asyncio
import json

import pytest

import kernel_env.client as module
from kernel_env.client import KernelEnv
from kernel_env.server.validation_memory import input_fingerprint,tensor_chunks

torch=pytest.importorskip('torch')


def test_mixed_dtype_inputs_all_bytes_and_cancelling_mutations():
    values=[torch.arange(23).to(dtype) for dtype in
            (torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.int8,torch.int64,torch.complex64)]
    values += [torch.tensor([False,True]),torch.arange(24).reshape(4,6).t(),torch.empty(0)]
    before=input_fingerprint(torch,values,max_bytes=16)
    for tensor in values:
        for i in range(tensor.numel()):
            copy=tensor.clone()
            index=[];flat=i
            for size in reversed(tensor.shape):index.insert(0,flat%size);flat//=size
            key=tuple(index)
            if tensor.dtype==torch.bool:tensor[key]=not tensor[key]
            else:tensor[key]+=1
            assert before!=input_fingerprint(torch,values,max_bytes=16),(tensor.dtype,i)
            tensor.copy_(copy)
    # Equal sums, squared sums and XOR summaries can collide: every byte still participates.
    a=torch.tensor([1.,2.,3.,4.,5.,6.]); b=a.clone();b[[0,1]]=b[[1,0]]
    assert a.sum()==b.sum() and (a*a).sum()==(b*b).sum()
    assert input_fingerprint(torch,[a],16)!=input_fingerprint(torch,[b],16)


def test_strided_broadcast_and_multiaxis_views_remain_bounded():
    tensors=[torch.arange(3*7*11).reshape(3,7,11).permute(2,0,1),
             torch.arange(10)[::2],torch.tensor([1.]).expand(8,9)]
    for x in tensors:
        chunks=list(tensor_chunks(x,32))
        assert sum(t.numel() for t in chunks)==x.numel()
        assert max(t.numel()*t.element_size() for t in chunks)<=32
        assert input_fingerprint(torch,[x],32)==input_fingerprint(torch,[x],32)


def test_reset_is_sent_once_after_connection_retries_and_preserves_sample(monkeypatch):
    sent=[];attempts=[]
    class Socket:
        async def send(self,payload):sent.append(json.loads(payload))
        async def recv(self):
            return json.dumps({'type':'reset','data':{'observation':{'stage':'ready','problem':'x'},'done':False,'reward':0}})
    async def connect(*args,**kwargs):
        attempts.append(1)
        if len(attempts)<3:raise TimeoutError('opening handshake')
        return Socket()
    async def sleep(_):pass
    monkeypatch.setattr(module,'ws_connect',connect);monkeypatch.setattr(module.asyncio,'sleep',sleep)
    client=KernelEnv(base_url='http://127.0.0.1:8000')
    asyncio.run(client.reset(task_id='level1/19_ReLU',split='train',seed=123))
    assert len(attempts)==3
    assert sent==[{'type':'reset','data':{'task_id':'level1/19_ReLU','split':'train','seed':123}}]
    client._ws=None


def test_lost_reset_response_is_not_retried_or_resampled(monkeypatch):
    sent=[]
    class Socket:
        async def send(self,payload):sent.append(json.loads(payload))
        async def recv(self):raise TimeoutError('response lost after server created episode')
    client=KernelEnv(base_url='http://127.0.0.1:8000');client._ws=Socket()
    with pytest.raises(TimeoutError):asyncio.run(client.reset(task_id='level1/19_ReLU',seed=123))
    assert len(sent)==1 and sent[0]['data']['seed']==123
    client._ws=None

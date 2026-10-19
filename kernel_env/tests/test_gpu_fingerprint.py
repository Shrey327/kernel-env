"""Paid validation gate for the actual GPU memory bound, not just CPU behavior."""
import time
import pytest
torch=pytest.importorskip('torch')
if not torch.cuda.is_available():pytest.skip('CUDA required',allow_module_level=True)
from kernel_env.server.validation_memory import input_fingerprint


def test_gpu_fingerprint_memory_is_bounded_and_sum_preserving_mutation_detected():
    x=torch.zeros(256*1024*1024,device='cuda',dtype=torch.float32)  # 1 GiB input
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();before=input_fingerprint(torch,[x]);seconds=time.monotonic()-start
    extra=torch.cuda.max_memory_allocated()-baseline
    assert extra<128*1024*1024,extra
    x[0]=1.;x[-1]=-1.
    assert before!=input_fingerprint(torch,[x])
    print({'input_gib':1,'fingerprint_seconds':seconds,'peak_extra_mib':extra/1024**2})

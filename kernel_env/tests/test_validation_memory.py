import pytest

torch = pytest.importorskip('torch')

from kernel_env.server.validation_memory import input_fingerprint, tensor_chunks, paired_tensor_chunks, max_abs_error


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16, torch.int64])
def test_fingerprint_checks_every_chunk_and_balanced_mutations(dtype):
    x = torch.arange(37).to(dtype)
    original = input_fingerprint(torch, [x], max_bytes=16)
    assert original == input_fingerprint(torch, [x.clone()], max_bytes=16)
    x[0] += 1
    x[-1] -= 1  # The previous global-sum mutation check missed this.
    assert original != input_fingerprint(torch, [x], max_bytes=16)


def test_noncontiguous_chunks_are_bounded_and_mutations_are_detected():
    x = torch.arange(60).reshape(5, 12).t()
    chunks = list(tensor_chunks(x, max_bytes=32))
    assert sum(c.numel() for c in chunks) == x.numel()
    assert all(c.numel() * c.element_size() <= 32 for c in chunks)
    before = input_fingerprint(torch, [x], max_bytes=32)
    assert before == input_fingerprint(torch, [x], max_bytes=32)
    x[-1, -1] += 1
    assert before != input_fingerprint(torch, [x], max_bytes=32)


def test_empty_scalar_and_nan_inputs_have_stable_byte_fingerprints():
    inputs = [torch.empty(0), torch.tensor(3.), torch.tensor([float('nan'), -0.0])]
    assert input_fingerprint(torch, inputs) == input_fingerprint(torch, inputs)


def test_fingerprint_preserves_raw_nan_payloads_and_signed_zero():
    import hashlib
    x=torch.tensor([0x7fc00001,0x7fc00002,-2147483648,0],dtype=torch.int32).view(torch.float32)
    expected=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert input_fingerprint(torch,[x],max_bytes=4)[0][2]==expected


@pytest.mark.parametrize('layout',['contiguous','transposed','expanded','scalar','empty'])
def test_matching_chunks_bound_both_layouts_and_cover_every_element(layout):
    reference=torch.arange(60,dtype=torch.float32).reshape(5,12)
    if layout=='transposed':reference=reference.t()
    elif layout=='expanded':reference=reference[:1].expand(5,12)
    elif layout=='scalar':reference=torch.tensor(3.)
    elif layout=='empty':reference=torch.empty(0)
    actual=reference.contiguous().clone()
    chunks=list(paired_tensor_chunks(reference,actual,max_bytes=16))
    assert sum(a.numel() for a,b in chunks)==reference.numel()
    assert all(a.numel()*a.element_size()<=16 and b.numel()*b.element_size()<=16 for a,b in chunks)
    assert all(torch.equal(a,b) for a,b in chunks)
    assert max_abs_error(torch,reference,actual,max_bytes=16)==0.
    if actual.numel():
        actual.reshape(-1)[-1]+=2
        assert max_abs_error(torch,reference,actual,max_bytes=16)==2.


def test_chunked_max_error_keeps_nonfinite_behavior_and_rejects_wrong_shape():
    assert max_abs_error(torch,torch.tensor([float('nan')]),torch.tensor([float('nan')]),4)==float('inf')
    with pytest.raises(ValueError,match='shape mismatch'):
        list(paired_tensor_chunks(torch.zeros(2),torch.zeros(3),4))

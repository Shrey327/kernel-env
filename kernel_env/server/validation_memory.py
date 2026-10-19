"""Bounded-memory validation; full input bytes participate in mutation checks."""
import hashlib

CHUNK_BYTES = 16 * 1024 * 1024


def paired_tensor_chunks(reference, actual, max_bytes=CHUNK_BYTES):
    """Matching bounded views without flattening a complete strided tensor."""
    if reference.shape != actual.shape:
        raise ValueError('comparison shape mismatch')
    if max_bytes < max(reference.element_size(), actual.element_size()):
        raise ValueError('comparison chunk cannot hold one element')
    size = reference.numel() * max(reference.element_size(), actual.element_size())
    if size <= max_bytes:
        yield reference, actual
    elif reference.is_contiguous() and actual.is_contiguous():
        left, right = reference.view(-1), actual.view(-1)
        count = max_bytes // max(reference.element_size(), actual.element_size())
        for start in range(0, left.numel(), count):
            length = min(count, left.numel() - start)
            yield left.narrow(0, start, length), right.narrow(0, start, length)
    else:
        dim = next(i for i, size in enumerate(reference.shape) if size > 1)
        width = max(1, reference.shape[dim] // 2)
        for start in range(0, reference.shape[dim], width):
            length = min(width, reference.shape[dim] - start)
            yield from paired_tensor_chunks(reference.narrow(dim, start, length),
                                            actual.narrow(dim, start, length), max_bytes)


def max_abs_error(torch, reference, actual, max_bytes=CHUNK_BYTES):
    maximum = 0.0
    for ref, out in paired_tensor_chunks(reference, actual, max_bytes):
        if ref.numel():
            value = (out.float() - ref.float()).abs().nan_to_num(float('inf')).max().item()
            maximum = max(maximum, value)
    return maximum


def tensor_chunks(tensor, max_bytes=CHUNK_BYTES):
    if tensor.numel() * tensor.element_size() <= max_bytes:
        yield tensor
    elif tensor.is_contiguous():
        flat = tensor.view(-1)
        count = max(1, max_bytes // tensor.element_size())
        for start in range(0, flat.numel(), count):
            yield flat.narrow(0, start, min(count, flat.numel() - start))
    else:
        # Splitting views first avoids materializing a full noncontiguous tensor.
        dim = next(i for i, size in enumerate(tensor.shape) if size > 1)
        width = max(1, tensor.shape[dim] // 2)
        for start in range(0, tensor.shape[dim], width):
            yield from tensor_chunks(tensor.narrow(dim, start, min(width, tensor.shape[dim] - start)), max_bytes)


def input_fingerprint(torch, inputs, max_bytes=CHUNK_BYTES):
    result = []
    for value in inputs:
        if not isinstance(value, torch.Tensor):
            continue
        digest = hashlib.sha256()
        for chunk in tensor_chunks(value.detach(), max_bytes):
            flat=chunk.reshape(-1)
            # Single-element views can be "contiguous" while retaining a non-unit stride.
            if flat.stride(0)!=1:flat=flat.clone(memory_format=torch.contiguous_format)
            raw = flat.view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        result.append((tuple(value.shape), str(value.dtype), digest.hexdigest()))
    return result

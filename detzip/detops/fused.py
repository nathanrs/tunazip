from functools import cache


@cache
def kernel(source, outputs=1):
    from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
    if outputs == 1:
        return _create_jit_fn(source)
    return _create_multi_output_jit_fn(source, num_outputs=outputs)


def literal(value):
    return float(value).hex()

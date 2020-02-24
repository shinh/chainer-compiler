from   chainer_compiler.elichika.typing.types import *

__all__ = [ 'check_dtype'
          ]

def check_dtype(module, dtype):
    for m in module.parameters():
        assert torch_dtype_to_np_dtype(m.dtype) == dtype, \
                "dtype mismatch in {}".format(module.__name__)
        # Checking the first param is enough
        return

import math
import numpy as np

from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.ext.common        import *
from   chainer_compiler.elichika.typing.ext.utils         import *
from   chainer_compiler.elichika.typing.types             import *
from   chainer_compiler.elichika.typing.ext.pytorch.utils import *

__all__ = [ 'ty_TorchIsTensor'
          , 'ty_TorchIdentical'
          , 'ty_TorchTensor'
          , 'ty_TorchTensorOfShape'
          , 'ty_TorchFromNumpy'
          , 'ty_TorchCat'
          , 'ty_TorchChunk'
          , 'ty_TorchExpand'
          , 'ty_TorchNumpy'
          , 'ty_TorchReshape'
          , 'ty_TorchSize'
          , 'ty_TorchSplit'
          , 'ty_TorchSqueeze'
          , 'ty_TorchStack'
          , 'ty_TorchTranspose'
          , 'ty_TorchUnsqueeze'
          , 'ty_TorchArith'
          , 'ty_TorchFlatten'
          , 'ty_TorchView'
          , 'ty_TorchRepeat'
          ]


class ty_TorchIsTensor():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        return TyBool(isinstance(x_type, TyTensor) and x_type.is_torch_tensor())


class ty_TorchIdentical():
    def __init__(self, is_float_only=True, ndim_min=None):
        self.is_float_only = is_float_only
        self.ndim_min = ndim_min

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        if self.is_float_only:
            assert x_type.dtype.kind == 'f'
        if self.ndim_min:
            assert x_type.ndim >= self.ndim_min
        return copy_ty(x_type)

    def nn(self, obj, ty_args, ty_kwargs):
        check_dtype(obj, ty_args[0].dtype)
        return self(ty_args, ty_kwargs)

# Tensors
## Creation Ops

class ty_TorchTensor(ty_MakeTensor):
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        # TODO(momohatt): Use global default dtype
        dtype = self.get_element_dtype(x_type)
        return TyTorchTensor(dtype, shape=self.calculate_shape(x_type))


class ty_TorchTensorOfShape():
    def __call__(self, ty_args, ty_kwargs):
        for ty in ty_args:
            unify(ty, TyInt())

        # TODO: use global default dtype
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float32'))
        assert not lacks_dtype

        shape = wrap_shape([extract_value_from_ty(ty) for ty in ty_args])
        return TyTorchTensor(dtype, shape=shape)


class ty_TorchFromNumpy():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.is_ndarray()
        x_type.kind = TensorKind.torch_tensor
        # XXX: Share reference of shape
        return TyTorchTensor(x_type.dtype, shape=x_type.shape)


## Indexing, Slicing, Joining, Mutating Ops

class ty_TorchCat():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, TyList)
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        x_type = xs_type.ty
        return TyTorchTensor(x_type.dtype, ndim=x_type.ndim)


class ty_TorchChunk():
    def __call__(self, ty_args, ty_kwargs):
        x_type, chunk_type = ty_args
        assert isinstance(chunk_type, TyNum)
        chunks = chunk_type.value

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)
        assert not lacks_dim
        return self.infer_return(x_type, chunks)

    def infer_return(self, x_type, chunks):
        ret_shape = list(x_type.shape)
        if chunks is None:
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        # TODO(momohatt): Handle cases where dim is not divisible by chunks
        ret_shape[self.dim] = ret_shape[self.dim] // chunks
        return TyTuple([TyTorchTensor(x_type.dtype, shape=ret_shape)
            for _ in range(chunks)])


class ty_TorchExpand():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert all([isinstance(ty, TyNum) for ty in ty_args[1:]])
        shape = [ty.value for ty in ty_args[1:]]
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchNumpy():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.is_torch_tensor()
        return TyNdarray(x_type.dtype, x_type.shape)


class ty_TorchReshape():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args
        assert isinstance(shape_type, TyTuple)
        assert shape_type.is_fixed_len

        self.shape = extract_value_from_ty(shape_type)
        return self.infer_return(x_type)

    def infer_return(self, x_type):
        ret_shape = calculate_reshape(x_type.shape, self.shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchSize():
    def __call__(self, ty_args, ty_kwargs):
        if len(ty_args) == 2:
            x_type, index_type = ty_args
            assert isinstance(index_type, TyNum)
            assert index_type.is_int()
            if index_type.value is None:
                return TyInt()
            return TyInt(x_type.shape[index_type.value].value)

        x_type, = ty_args
        return TyTuple([TyInt(e.value) for e in x_type.shape])


class ty_TorchSplit():
    def __call__(self, ty_args, ty_kwargs):
        x_type, split_size_or_sections_type = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        # TODO: Handle cases where lacks_dim = True

        if isinstance(split_size_or_sections_type, TyNum):
            size = split_size_or_sections_type.value
            assert size is None or size > 0
            return self.infer_return_size(x_type, size)

        sections_type = split_size_or_sections_type
        assert isinstance(sections_type, TyTuple)
        return self.infer_return_sections(x_type, sections_type)

    def infer_return_size(self, x_type, size):
        if size is None:
            if self.dim is None:
                return TyTuple(TyTorchTensor(x_type.dtype, ndim=x_type.ndim))
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        if x_type.shape[self.dim].is_null():
            # TODO
            pass

        n_split = math.ceil(x_type.shape[self.dim].get_value() / size)
        if x_type.shape[self.dim] % size != 0:
            ret_shapes = [list(x_type.shape) for _ in range(n_split)]
            for i in range(n_split - 1):
                ret_shapes[i][self.dim] = size
            ret_shapes[-1][self.dim] = x_type.shape[self.dim] % size
            return TyTuple(
                    [TyTorchTensor(x_type.dtype, shape=shape) for shape in ret_shapes])

        ret_shape = list(x_type.shape)
        ret_shape[self.dim] = size
        return TyTuple(
            [TyTorchTensor(x_type.dtype, shape=ret_shape)] * n_split)

    def infer_return_sections(self, x_type, sections_type):
        if not sections_type.is_fixed_len:
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        sections = extract_value_from_ty(sections_type)
        ret_shapes = [list(x_type.shape) for _ in sections]
        for i, n in enumerate(sections):
            ret_shapes[i][self.dim] = n
        return TyTuple([TyTorchTensor(x_type.dtype, shape=shape)
            for shape in ret_shapes])


class ty_TorchSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)

        if is_incomplete_shape(x_type.shape):
            if lacks_dim or self.dim is None:
                assert False, "torch.squeeze: cannot guess ndim of return type"

        return self.infer_return(x_type)

    def infer_return(self, x_type):
        if self.dim is not None:
            ret_shape = remove_dims(x_type.shape, (self.dim,))
        else:
            ret_shape = [s for s in x_type.shape if s != 1]
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchStack():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, TyList)

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        if lacks_dim:
            x_type = xs_type.ty
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        self.dim %= xs_type.ty.ndim + 1
        return self.infer_return(xs_type)

    def infer_return(self, xs_type):
        ret_shape = list(xs_type.ty.shape)
        ret_shape.insert(self.dim, ShapeElem(None))
        return TyTorchTensor(xs_type.ty.dtype, shape=ret_shape)


class ty_TorchTranspose():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dim1_type, dim2_type = ty_args
        assert isinstance(dim1_type, TyNum)
        assert isinstance(dim2_type, TyNum)
        dim1 = dim1_type.value
        dim2 = dim2_type.value
        assert dim1 is not None and dim2 is not None
        shape = list(x_type.shape)
        shape[dim1], shape[dim2] = x_type.shape[dim2], x_type.shape[dim1]
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchUnsqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dim_type = ty_args
        assert isinstance(dim_type, TyNum)
        dim = extract_value_from_ty(dim_type)
        if dim is None:
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        shape = list(x_type.shape)
        shape.insert(dim, 1)
        return TyTorchTensor(x_type.dtype, shape=shape)


# Math operations
## Pointwise Ops

class ty_TorchArith(ty_TensorArith):
    def __init__(self, op):
        super().__init__(TensorKind.torch_tensor, op)


## Other operations

class ty_TorchFlatten():
    def __call__(self, ty_args, ty_kwargs):
        input_type, = ty_args
        shape = input_type.shape
        start_dim, _ = get_kwarg(ty_kwargs, 'start_dim', default=0)
        end_dim, _   = get_kwarg(ty_kwargs, 'end_dim', default=-1)

        prefix_shape = shape[:start_dim]
        middle_shape = shape[start_dim:end_dim] + (shape[end_dim],)
        postfix_shape = shape[end_dim:][2:]
        size = size_of_shape(middle_shape)
        out_shape = prefix_shape + (size,) + postfix_shape
        return TyTorchTensor(input_type.dtype, shape=out_shape)


# torch.Tensor

class ty_TorchView():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        shape_type = ty_args[1:]
        assert isinstance(x_type, TyTensor)

        out_shape = wrap_shape([extract_value_from_ty(t) for t in shape_type])
        ret_shape = calculate_reshape(x_type.shape, out_shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchRepeat():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        size_types = ty_args[1:]
        assert len(size_types) >= x_type.ndim
        assert all([isinstance(ty, TyNum) for ty in size_types])
        sizes = [ty.value for ty in size_types]
        return self.infer_return(x_type, sizes)

    def infer_return(self, x_type, sizes):
        n = len(sizes) - x_type.ndim
        for i in range(n, len(sizes)):
            sizes[i] *= x_type.shape[i - n]
        return TyTorchTensor(x_type.dtype, shape=sizes)

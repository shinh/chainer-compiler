import torch
import torch.nn as nn
import torch.nn.functional as F

from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.ext.utils          import *
from   chainer_compiler.elichika.typing.types              import *
from   chainer_compiler.elichika.typing.ext.common         import *
from   chainer_compiler.elichika.typing.ext.pytorch.nn     import *
from   chainer_compiler.elichika.typing.ext.pytorch.tensor import *

__all__ = [ 'pytorch_attr_ty', 'pytorch_func_ty', 'pytorch_callable_ty' ]


pytorch_attr_ty = {
        'shape' : ty_Shape,
        'dtype' : ty_DType,
        }


pytorch_func_ty = {
        torch.is_tensor  : ty_TorchIsTensor(),

        # https://pytorch.org/docs/stable/torch.html#creation-ops
        torch.tensor     : ty_TorchTensor(),
        torch.zeros      : ty_TorchTensorOfShape(),
        torch.ones       : ty_TorchTensorOfShape(),
        torch.rand       : ty_TorchTensorOfShape(),
        torch.randn      : ty_TorchTensorOfShape(),
        torch.from_numpy : ty_TorchFromNumpy(),

        # https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
        torch.cat       : ty_TorchCat(),
        torch.chunk     : ty_TorchChunk(),
        torch.reshape   : ty_TorchReshape(),
        torch.split     : ty_TorchSplit(),
        torch.squeeze   : ty_TorchSqueeze(),
        torch.stack     : ty_TorchStack(),
        torch.transpose : ty_TorchTranspose(),
        torch.unsqueeze : ty_TorchUnsqueeze(),

        # https://pytorch.org/docs/stable/torch.html#random-sampling
        torch.rand_like  : ty_TorchIdentical(),
        torch.randn_like : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/torch.html#math-operations
        torch.abs     : ty_TorchIdentical(),
        torch.cos     : ty_TorchIdentical(),
        torch.cosh    : ty_TorchIdentical(),
        torch.exp     : ty_TorchIdentical(),
        torch.log     : ty_TorchIdentical(),
        torch.sigmoid : ty_TorchIdentical(),
        torch.sin     : ty_TorchIdentical(),
        torch.sinh    : ty_TorchIdentical(),
        torch.sqrt    : ty_TorchIdentical(),
        torch.tan     : ty_TorchIdentical(),
        torch.tanh    : ty_TorchIdentical(),

        torch.add     : ty_TorchArith(),
        torch.sub     : ty_TorchArith(),
        torch.mul     : ty_TorchArith(),

        torch.flatten : ty_TorchFlatten(),

        # https://pytorch.org/docs/stable/nn.functional.html#pooling-functions
        F.avg_pool1d  : ty_TorchPooling(dim=1),
        F.avg_pool2d  : ty_TorchPooling(dim=2),
        F.avg_pool3d  : ty_TorchPooling(dim=3),
        F.max_pool1d  : ty_TorchPooling(dim=1),
        F.max_pool2d  : ty_TorchPooling(dim=2),
        F.max_pool3d  : ty_TorchPooling(dim=3),

        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        F.relu        : ty_TorchIdentical(),
        F.softmax     : ty_TorchIdentical(),
        F.log_softmax : ty_TorchIdentical(),
        F.tanh        : ty_TorchIdentical(),
        F.sigmoid     : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/nn.functional.html#sparse-functions
        F.embedding   : ty_TorchEmbed(),

        # https://pytorch.org/docs/stable/nn.functional.html#vision-functions
        F.interpolate : ty_TorchInterpolate(),

        torch.Tensor.add  : ty_TorchArith(),
        torch.Tensor.add_ : ty_TorchArith(),
        torch.Tensor.sub  : ty_TorchArith(),
        torch.Tensor.sub_ : ty_TorchArith(),
        torch.Tensor.mul  : ty_TorchArith(),
        torch.Tensor.mul_ : ty_TorchArith(),

        torch.Tensor.chunk     : ty_TorchChunk(),
        torch.Tensor.cpu       : ty_TorchIdentical(is_float_only=False),
        torch.Tensor.numpy     : ty_TorchNumpy(),
        torch.Tensor.repeat    : ty_TorchRepeat(),
        torch.Tensor.size      : ty_TorchSize(),
        torch.Tensor.squeeze   : ty_TorchSqueeze(),
        torch.Tensor.transpose : ty_TorchTranspose(),
        torch.Tensor.unsqueeze : ty_TorchUnsqueeze(),
        torch.Tensor.view      : ty_TorchView(),

        torch.Tensor.detach    : ty_TorchIdentical(is_float_only=False),
        }


pytorch_callable_ty = {
        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        nn.Conv1d            : ty_TorchConv(dim=1).nn,
        nn.Conv2d            : ty_TorchConv(dim=2).nn,
        nn.Conv3d            : ty_TorchConv(dim=3).nn,
        nn.ConvTranspose1d   : ty_TorchConv(dim=1, transpose=True).nn,
        nn.ConvTranspose2d   : ty_TorchConv(dim=2, transpose=True).nn,
        nn.ConvTranspose3d   : ty_TorchConv(dim=3, transpose=True).nn,

        # https://pytorch.org/docs/stable/nn.html#pooling-layers
        nn.AvgPool1d         : ty_TorchPooling(dim=1).nn,
        nn.AvgPool2d         : ty_TorchPooling(dim=2).nn,
        nn.AvgPool3d         : ty_TorchPooling(dim=3).nn,
        nn.MaxPool1d         : ty_TorchPooling(dim=1).nn,
        nn.MaxPool2d         : ty_TorchPooling(dim=2).nn,
        nn.MaxPool3d         : ty_TorchPooling(dim=3).nn,
        nn.AdaptiveAvgPool1d : ty_TorchAdaptivePooling(dim=1).nn,
        nn.AdaptiveAvgPool2d : ty_TorchAdaptivePooling(dim=2).nn,
        nn.AdaptiveAvgPool3d : ty_TorchAdaptivePooling(dim=3).nn,

        # https://pytorch.org/docs/stable/nn.html#padding-layers
        nn.ReflectionPad1d   : ty_TorchPad(dim=1).nn,
        nn.ReflectionPad2d   : ty_TorchPad(dim=2).nn,
        nn.ReplicationPad1d  : ty_TorchPad(dim=1).nn,
        nn.ReplicationPad2d  : ty_TorchPad(dim=2).nn,
        nn.ReplicationPad3d  : ty_TorchPad(dim=3).nn,
        nn.ZeroPad2d         : ty_TorchPad(dim=2).nn,
        nn.ConstantPad1d     : ty_TorchPad(dim=1, is_const=True).nn,
        nn.ConstantPad2d     : ty_TorchPad(dim=2, is_const=True).nn,
        nn.ConstantPad3d     : ty_TorchPad(dim=3, is_const=True).nn,

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        nn.LeakyReLU        : ty_TorchIdentical().nn,
        nn.ReLU             : ty_TorchIdentical().nn,
        nn.Sigmoid          : ty_TorchIdentical().nn,
        nn.Tanh             : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-other

        # https://pytorch.org/docs/stable/nn.html#normalization-layers
        nn.BatchNorm1d      : ty_TorchBatchNorm(dim=1).nn,
        nn.BatchNorm2d      : ty_TorchBatchNorm(dim=2).nn,
        nn.BatchNorm3d      : ty_TorchBatchNorm(dim=3).nn,
        nn.InstanceNorm1d   : ty_TorchInstanceNorm(dim=1).nn,
        nn.InstanceNorm2d   : ty_TorchInstanceNorm(dim=2).nn,
        nn.InstanceNorm3d   : ty_TorchInstanceNorm(dim=3).nn,

        # https://pytorch.org/docs/stable/nn.html#recurrent-layers
        nn.LSTMCell         : ty_TorchLSTMCell().nn,

        # https://pytorch.org/docs/stable/nn.html#linear-layers
        nn.Linear           : ty_TorchLinear().nn,

        # https://pytorch.org/docs/stable/nn.html#dropout-layers
        nn.Dropout          : ty_TorchIdentical().nn,
        nn.Dropout2d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.Dropout3d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.AlphaDropout     : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#sparse-layers
        nn.Embedding        : ty_TorchEmbed().nn,

        # https://pytorch.org/docs/stable/nn.html#loss-functions
        nn.CrossEntropyLoss : ty_TorchNNCrossEntropyLoss().nn,

        # https://pytorch.org/docs/stable/nn.html#vision-layers
        nn.PixelShuffle     : ty_TorchPixelShuffle().nn,
        }

import chainer
import chainer.functions as F
import chainer.links as L

import onnx
import onnx.helper as oh
from onnx import numpy_helper
from onnx import TensorProto
from onnx import ModelProto

import elichika.parser.core as core
import elichika.parser.graphs as graphs
import elichika.parser.values as values
import elichika.parser.nodes as nodes
import elichika.parser.functions as functions
import elichika.parser.functions_builtin as functions_builtin
import elichika.parser.utils as utils

import numpy as np
import collections

import elichika.onnx_converters as oc

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)

def _list(v) -> 'List[int]':
    if isinstance(v, collections.Iterable):
        return list(x for x in v)
    return [v]


def convert_relu(onnx_graph, node):
    onnx_graph.add_node('Relu',
                        [node.inputs[0]],
                        [node.outputs[0]],
                        name=str(node.lineprop))


def convert_softmax(onnx_graph, node):
    onnx_graph.add_node(
        "Softmax",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        axis=oc.try_get_attribute(node.inputs[1]))


def convert_pad_sequence(onnx_graph, node):
    kwargs = {}

    if node.inputs[1] is not None:
        value = oc.try_get_attribute(node.inputs[1])
        if value is not None:
            kwargs['length'] = value
        if node.inputs[2] is not None:
            value = oc.try_get_attribute(node.inputs[2])
            if value != 0:
                kwargs['value'] = float(value)

    onnx_graph.add_node(
        "ChainerSequencePad",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        **kwargs)


def convert_dropout(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    ratio = oc.try_get_attribute(node.args.keywords['ratio'])

    onnx_graph.add_node(
        "Dropout",
        [x],
        node.outputs,
        str(node.lineprop),
        ratio=ratio,
        )


def convert_matmul(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('a', oc.ParseType.In)
    parser.add_def('b', oc.ParseType.In)
    parser.add_def('transa', oc.ParseType.Att, False)
    parser.add_def('transb', oc.ParseType.Att, False)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "MatMul",
        [parser.get('a').create_tensor(), parser.get('b').create_tensor()],
        node.outputs,
        str(node.lineprop),
        )

def convert_concat(onnx_graph, node):
    xs = oc.ONNXValue(onnx_graph,node.args.keywords['xs'])
    axis = oc.try_get_attribute(node.args.keywords['axis'])

    if isinstance(node.args.inputs[0], values.TupleValue) and node.args.inputs[0].has_constant_value():
        vs = []
        for v in xs.value.get_constant_value():
            v_ = oc.ONNXValue(onnx_graph, v)
            vs.append(v_)

        onnx_graph.add_node(
            "Concat",
            vs,
            node.outputs,
            str(node.lineprop),
            axis=axis,
            )

    elif isinstance(node.args.inputs[0], values.ListValue):
        onnx_graph.add_node(
            "ChainerSequenceConcat",
            [xs.create_sequence()],
            node.outputs,
            str(node.lineprop),
            axis=axis,
            )

def convert_softmax_cross_entropy(onnx_graph, node):
    normalize = oc.try_get_attribute(node.args.keywords['normalize'])
    cache_score = oc.try_get_attribute(node.args.keywords['cache_score'])
    class_weight = oc.try_get_attribute(node.args.keywords['class_weight'])
    ignore_label = oc.try_get_attribute(node.args.keywords['ignore_label'])
    reduce = oc.try_get_attribute(node.args.keywords['reduce'])
    enable_double_backprop = oc.try_get_attribute(node.args.keywords['enable_double_backprop'])

    assert normalize  # TODO(hamaji): Not supported yet.
    assert cache_score  # TODO(hamaji): Not supported yet.
    assert class_weight is None  # TODO(hamaji): Not supported yet.
    assert ignore_label == -1  # TODO(hamaji): Not supported yet.
    assert reduce == 'mean'  # TODO(hamaji): Not supported yet.
    assert not enable_double_backprop  # TODO(hamaji): Not supported yet.

    onnx_graph.add_node(
        "ChainerSoftmaxCrossEntropy",
        node.inputs[0:2],
        node.outputs,
        str(node.lineprop))

def convert_max_pooling_2d(onnx_graph, node):
    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    ksize = oc.try_get_attribute(node.args.keywords['ksize'])
    stride = oc.try_get_attribute(node.args.keywords['stride'])
    pad = oc.try_get_attribute(node.args.keywords['pad'])
    cover_all = oc.try_get_attribute(node.args.keywords['cover_all'])
    return_indices = oc.try_get_attribute(node.args.keywords['return_indices'])

    assert not return_indices  # TODO(hamaji): Not implemented yet.

    kwargs = {}
    kwargs['kernel_shape'] = _pair(ksize)

    if stride is not None:
        kwargs['strides'] = _pair(stride)
    else:
        kwargs['strides'] = _pair(ksize)

    if pad is not None:
        kwargs['pads'] = _pair(pad) * 2
    else:
        kwargs['pads'] = _pair(0)

    onnx_graph.add_node(
        "MaxPool",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        chainer_cover_all=cover_all,
        **kwargs,
        )

def convert_average_pool_2d(onnx_graph, node):
    kwargs = {}
    ksize = oc.try_get_attribute(node.inputs[1])
    kwargs['kernel_shape'] = _pair(ksize)

    value = oc.try_get_attribute(node.inputs[2])
    if value is not None:
        kwargs['strides'] = _pair(value)
    else:
        kwargs['strides'] = _pair(ksize)

    value = oc.try_get_attribute(node.inputs[3])
    if value is not None:
        kwargs['pads'] = _pair(value) * 2
    else:
        kwargs['pads'] = _pair(0)

    kwargs['count_include_pad'] = 1

    onnx_graph.add_node(
        "AveragePool",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        **kwargs,
        )

def convert_unpooling_2d(onnx_graph, node : 'nodes.NodeCall'):
    ksize = oc.try_get_attribute(node.args.keywords['ksize'])
    stride = oc.try_get_attribute(node.args.keywords['stride'])
    pad = oc.try_get_attribute(node.args.keywords['pad'])
    outsize = oc.try_get_attribute(node.args.keywords['outsize'])
    cover_all = oc.try_get_attribute(node.args.keywords['cover_all'])

    assert(stride is None) # TODO(hamaji): Not supported yet.
    assert(pad == 0) # TODO(hamaji): Not supported yet.
    assert(outsize is None) # TODO(hamaji): Not supported yet.
    assert(cover_all is False) # TODO(hamaji): Not supported yet.

    scales = np.array([1, 1] + list(_pair(ksize)), dtype=np.float32)
    scales_ = oc.ONNXValue(onnx_graph, scales, [node, '/Scale'], is_constant = True)
    onnx_graph.add_node(
        "Upsample",
        [node.inputs[0], scales_],
        [node.outputs[0]],
        name=str(node.lineprop))

def convert_resize_images(onnx_graph, node):
    output_shape = oc.try_get_attribute(node.args.keywords['output_shape'])

    onnx_graph.add_node(
        "ChainerResizeImages",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        output_shape=_pair(output_shape))

def convert_vstack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceConcat",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=0)

def convert_hstack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceConcat",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=1)

def convert_stack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceStack",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=parser.get('axis'))

def convert_separate(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceSeparate",
        [parser.get('x').create_tensor()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=parser.get('axis'))

def convert_squeeze(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    kwargs = {}
    if parser.get('axis') is not None:
        kwargs['axes'] = _list(parser.get('axis'))

    onnx_graph.add_node(
        "Squeeze",
        [parser.get('x').create_tensor()],
        [node.outputs[0]],
        name=str(node.lineprop),
        **kwargs)

def convert_reshape(onnx_graph, node):
    onnx_graph.add_node(
        "Reshape",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.inputs[1]).create_tensor()],
        node.outputs,
        str(node.lineprop))

def convert_split_axis(onnx_graph, node):
    force_tuple = oc.try_get_attribute(node.args.keywords['force_tuple'])
    assert(force_tuple is True) # TODO(hamaji): Not supported yet.

    onnx_graph.add_node(
        "ChainerSequenceSplitAxis",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.args.keywords['indices_or_sections']).create_tensor()],
        node.outputs,
        str(node.lineprop),
        axis = oc.try_get_attribute(node.args.keywords['axis']))

def convert_swapaxes(onnx_graph, node):
    axis1 = oc.try_get_attribute(node.args.keywords['axis1'])
    axis2 = oc.try_get_attribute(node.args.keywords['axis2'])
    pe = list(range(max(axis1, axis2)+1))
    pe[axis1] = axis2
    pe[axis2] = axis1

    onnx_graph.add_node(
        "Transpose",
        [node.inputs[0]],
        node.outputs,
        str(node.lineprop),
        perm = pe)

def convert_roi_max_pooling_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIMaxPool2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value))
    return

def convert_roi_average_pooling_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIAveragePool2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value))
    return

def convert_roi_max_align_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])
    sampling_ratio = oc.ONNXValue(onnx_graph,node.args.keywords['sampling_ratio'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIMaxAlign2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value),
        sampling_ratio=_pair(oc.try_get_attribute(sampling_ratio.value)))
    return

def convert_roi_average_align_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])
    sampling_ratio = oc.ONNXValue(onnx_graph,node.args.keywords['sampling_ratio'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIAverageAlign2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value),
        sampling_ratio=_pair(oc.try_get_attribute(sampling_ratio.value)))
    return

def convert_local_response_normalization(onnx_graph, node):
    kwargs = {}
    kwargs['size'] = oc.try_get_attribute(node.args.keywords['n'])
    kwargs['bias'] = float(oc.try_get_attribute(node.args.keywords['k']))
    kwargs['alpha'] = float(oc.try_get_attribute(node.args.keywords['alpha']) * kwargs['size'])
    kwargs['beta'] = float(oc.try_get_attribute(node.args.keywords['beta']))

    onnx_graph.add_node(
        "LRN",
        [node.inputs[0]],
        node.outputs,
        str(node.lineprop),
        **kwargs,
    )

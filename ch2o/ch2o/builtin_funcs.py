# coding: utf-8

from onnx import helper

from ch2o.callable import Callable
from ch2o.funcs import Func, totensor
from ch2o.utils import istensor, new_tensor


class Builtin_Len(Callable):
    def __init__(self):
        super(Builtin_Len, self).__init__(lambda x: x)

    def call_impl(self, env, x):
        x = x.to_tensor(env)
        res = new_tensor()
        v = new_tensor()
        env.addnode(
            "Shape",
            inputs=[x.name], outputs=[v.name]
        )

        env.addnode(
            "Gather",
            inputs=[v.name, totensor(0, env).name], outputs=[res.name],
            axis=0
        )

        return res


def builtin_range(args, _, env):
    if all(a.is_py for a in args):
        # print('constant loop',args)
        return range(*(a.value for a in args))

    assert len(args) <= 1  # TODO(satos) 一般的にする

    a = new_tensor()
    b = new_tensor()
    res = new_tensor()
    env.addnode(
        'Loop',
        inputs=[args[0].to_tensor(env).name, ""], outputs=[res.name],
        body=helper.make_graph(
            [],
            "Range_subgraph",
            [a, b], [b, a]
        )
    )
    return res


builtin_functions = {
    'len': Builtin_Len(),
    'range': Func(builtin_range),
}

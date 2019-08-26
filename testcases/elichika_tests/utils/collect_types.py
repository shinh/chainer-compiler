# Usage:
#
# $ PYTHONPATH=. python3 testcases/elichika_tests/utils/collect_types.py

import importlib
import inspect
import os
import sys
import types

import chainer
import numpy

from chainer_compiler.elichika import testtools


def _run_model(model, args, subname=None, backprop=False):
    if (isinstance(model, type) or
        isinstance(model, types.FunctionType)):
        model = model()
    model(*args)


testtools.generate_testcase = _run_model


show_shape_info = False


class TypeInfo(object):
    def __init__(self, value):
        self.type = type(value)
        self.dtype = None
        self.shape = None
        if isinstance(value, (numpy.ndarray, chainer.Variable)):
            self.dtype = value.dtype
            if show_shape_info:
                self.shape = value.shape

    def __str__(self):
        r = self.type.__name__
        if self.dtype is not None:
            p = str(self.dtype)
            if self.shape is not None:
                p += ', {}'.format(self.shape)
            r += '[{}]'.format(p)
        return r


class TypeCollector(object):
    def __init__(self):
        self.wrapped_functions = {}
        self.call_records = []

        self.hook_funcs('chainer.functions', chainer.functions)
        self.hook_funcs('numpy', numpy)

    def hook_funcs(self, module_name, module):
        for func_name in sorted(dir(module)):
            func = getattr(module, func_name)
            if isinstance(func, types.FunctionType):
                wrap_func = self.make_wrap_func(func)
                self.wrapped_functions[func] = (module_name, func_name)
                setattr(module, func_name, wrap_func)

    def make_wrap_func(self, func):
        try:
            sig = inspect.signature(func)
        except ValueError:
            sig = None

        def wrap_func(*args, **kwargs):
            ret = func(*args, **kwargs)

            # Try canonicalization.
            if sig is not None:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                args = bound.args
                kwargs = bound.kwargs

            arg_types = [TypeInfo(a) for a in args]
            kwarg_types = {k: TypeInfo(a) for k, a in kwargs.items()}
            if isinstance(ret, (list, tuple)):
                ret_types = [TypeInfo(a) for a in ret]
            else:
                ret_types = TypeInfo(ret)
            self.call_records.append((func, arg_types, kwarg_types, ret_types))

            return ret

        return wrap_func

    def dump_typeinfo(self):
        typeinfo_by_func = {}
        for func, arg_types, kwarg_types, ret_types in self.call_records:
            arg_types = tuple(arg_types)
            kwarg_types = tuple((k, kwarg_types[k])
                                for k in sorted(kwarg_types))
            if isinstance(ret_types, list):
                ret_types = tuple(ret_types)

            if func not in typeinfo_by_func:
                typeinfo_by_func[func] = set()
            typeinfo_by_func[func].add((arg_types, kwarg_types, ret_types))

        for func, (module_name, func_name) in self.wrapped_functions.items():
            if func not in typeinfo_by_func:
                continue

            name = '{}.{}'.format(module_name, func_name)
            sigs = set()
            for arg_types, kwarg_types, ret_types in typeinfo_by_func[func]:
                args_strs = []
                for ti in arg_types:
                    args_strs.append(str(ti))
                for k, ti in kwarg_types:
                    args_strs.append('{}: {}'.format(k, str(ti)))
                args_str = ', '.join(args_strs)

                if isinstance(ret_types, tuple):
                    ret_str = ', '.join(str(ti) for ti in ret_types)
                else:
                    ret_str = str(ret_types)
                sigs.add('{}({}) -> {}'.format(name, args_str, ret_str))

            for sig in sorted(sigs):
                print(sig)



def collect_test_filenames():
    all_tests = []
    for dirpath, _, filenames in os.walk('testcases/elichika_tests'):
        for filename in filenames:
            if 'canonicalizer' in dirpath:
                continue
            if dirpath.endswith('utils'):
                continue
            if filename.endswith('.py'):
                all_tests.append(os.path.join(dirpath, filename))
    return list(sorted(set(all_tests)))


def main():
    type_collector = TypeCollector()

    test_filenames = collect_test_filenames()
    for filename in test_filenames:
        module_name = os.path.splitext(filename)[0].replace('/', '.')
        module = importlib.import_module(module_name)

        if not hasattr(module, 'main'):
            continue

        print('Running', filename, '...')

        module.main()


    print('Type info:')
    type_collector.dump_typeinfo()



if __name__ == '__main__':
    main()

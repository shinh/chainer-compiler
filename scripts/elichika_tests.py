#!/usr/bin/python

import importlib
import glob
import os
import subprocess
import sys

from test_case import TestCase


class Generator(object):
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename


# TODO(hamaji): Triage failing tests.
TESTS = [
    # Generator('node', 'Convolution2D'),
    Generator('node', 'Linear'),
    # Generator('node', 'Relu'),
    # Generator('node', 'Softmax'),

    # Generator('syntax', 'ChinerFunctionNode'),
    # Generator('syntax', 'Cmp'),
    # Generator('syntax', 'For'),
    # Generator('syntax', 'ForAndIf'),
    # Generator('syntax', 'If'),
    # Generator('syntax', 'LinkInFor'),
    # Generator('syntax', 'ListComp'),
    Generator('syntax', 'MultiClass'),
    Generator('syntax', 'MultiFunction'),
    # Generator('syntax', 'Range'),
    # Generator('syntax', 'Sequence'),
    # Generator('syntax', 'Slice'),
    # Generator('syntax', 'UserDefinedFunc'),
]


def get_test_generators(dirname):
    return [test for test in TESTS if test.dirname == dirname]


def print_test_generators(dirname):
    tests = []
    for gen in get_test_generators(dirname):
        tests.append(
            os.path.join('elichika/tests', gen.dirname, gen.filename + '.py'))
    print(';'.join(tests))


def generate_tests(dirname):
    from testtools import testcasegen

    for gen in get_test_generators(dirname):
        py = os.path.join('tests', gen.dirname, gen.filename)
        out_dir = os.path.join('out', 'elichika_%s_%s' %
                               (gen.dirname, gen.filename))
        print('Running %s' % py)
        module = importlib.import_module(py.replace('/', '.'))
        testcasegen.reset_test_generator([out_dir])
        module.main()


if __name__ == '__main__':
    if sys.argv[1] == '--list':
        print_test_generators(sys.argv[2])
    elif sys.argv[1] == '--generate':
        generate_tests(sys.argv[2])
    else:
        raise RuntimeError('See %s for the usage' % sys.argv[0])

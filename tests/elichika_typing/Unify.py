from copy import deepcopy
import unittest

import chainer_compiler.elichika.parser.types as T


def is_unifiable(ty1, ty2):
    try:
        T.unify(ty1, ty2)
        return True
    except T.UnifyError as e:
        return False


class TestUnify(unittest.TestCase):
    def test_not_unifiable(self):
        x = T.TyVar()
        ty1 = T.TyArrow([x], x)
        ty2 = T.TyArrow([T.TyString()], T.TyInt())
        self.assertFalse(is_unifiable(ty1, ty2))

    def test_deepcopy(self):
        x = T.TyVar()
        ty = T.TyArrow([x], x)
        ty1 = deepcopy(ty)
        ty2 = T.TyArrow([T.TyString()], T.TyInt())
        self.assertFalse(is_unifiable(ty1, ty2))

    def test_union(self):
        x = T.TyVar()
        ty1 = T.TyUnion(
                T.TyArrow([T.TyIntOnly()], T.TyIntOnly()),
                T.TyArrow([T.TyFloat()], T.TyFloat()),
                )
        ty2 = T.TyArrow([x], T.TyFloat())
        self.assertTrue(is_unifiable(ty1, ty2))



def main():
    unittest.main()


if __name__ == '__main__':
    main()

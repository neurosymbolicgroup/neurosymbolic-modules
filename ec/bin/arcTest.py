import binutil
from dreamcoder.domains.arc.arcPrimitives import map_multiple, _equals_invariant, _equals_exact, _color_transform
from dreamcoder.domains.arc.arcPrimitives import Grid
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as pd
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
import numpy as np
import unittest


class TestArcPrimitives(unittest.TestCase):

    def test_map_multiple(self):
        a = np.array([[1, 2, 3]])
        b = map_multiple(a, [1,2], [2, 3])
        b_goal = [[2, 3, 3]]
        c = map_multiple(map_multiple(a, [1], [2]), [2], [3])
        c_goal = [[3, 3, 3]]
        self.assertTrue(b.tolist(), b_goal)
        self.assertEqual(c.tolist(), c_goal)

    def test_equals_exact(self):
        for i in range(10):
            a = np.random.randint(0, high=9, size=(3, 3))
            self.assertTrue(_equals_exact(Grid(a))(Grid(np.copy(a))))
        
        self.assertFalse(_equals_exact(Grid(np.eye(3)))(Grid(np.ones((3,3)))))

    def test_equals_invariance(self):
        a = Grid(np.array([[1]]))
        # size
        for i in range(1, 10):
            b = Grid(np.ones((i, i)))
            self.assertTrue(_equals_invariant(a)(b)('size'))
            self.assertTrue(_equals_invariant(b)(a)('size'))

            c = Grid(np.random.randint(0, high=9, size=(3, 1)))
            d = p._inflate(c)(i)
            self.assertTrue(_equals_invariant(c)(d)('size'))
            self.assertTrue(_equals_invariant(d)(c)('size'))
            self.assertFalse(_equals_invariant(a)(c)('size'))

        # rotation
        a = Grid(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        b = a
        for i in range(3):
            b = p._rotate_ccw(b)
            self.assertTrue(_equals_invariant(a)(b)('rotation'))

        self.assertFalse(_equals_invariant(a)(p._y_mirror(a))('rotation'))
        # color
        a = Grid(np.array([[1,1,2,3]]))
        b = Grid(np.array([[2,2,1,3]]))
        self.assertTrue(_equals_invariant(a)(b)('color'))
        a = Grid(np.array([[1,1,2,3]]))
        b = Grid(np.array([[2,2,3,4]]))
        self.assertTrue(_equals_invariant(a)(b)('color'))
        a = Grid(np.array([[1,1,2,3]]))
        b = Grid(np.array([[2,3,3,4]]))
        self.assertFalse(_equals_invariant(a)(b)('color'))

    def test_inflate(self):
        a = Grid(np.array([[1, 2]]))
        b = Grid(np.array([[1, 1, 2, 2], [1, 1, 2, 2]]))
        self.assertTrue(_equals_exact(p._inflate(a)(2))(b))

    def test_place_into_grid(self):
        example0 = task = get_arc_task(168).examples[0]
        inp, outp = example0[0][0], example0[1]
        inp = p._input(inp)
        objects = p._objects(inp)
        self.assertTrue(_equals_exact(inp)(
            p._place_into_input_grid(p._objects(inp))))

    def test_objects(self):
        example0 = task = get_arc_task(359).examples[0]
        inp, outp = example0[0][0], example0[1]
        self.assertEqual(len(p._objects2(p._input(inp))(False)(False)), 1)
        self.assertEqual(p._place_into_input_grid(p._objects2(p._input(inp))(False)(False)), p._input(inp))

        example0 = task = get_arc_task(78).examples[1]
        inp, outp = example0[0][0], example0[1]
        objects = p._objects2(p._input(inp))(False)(True)
        # no diagonal connect, yes separate colors
        self.assertEqual(len(p._objects2(p._input(inp))(False)(True)), 20)
        # diagonal connect, no separate colors
        self.assertEqual(len(p._objects2(p._input(inp))(True)(False)), 7)
        # diagonal connect, separate colors
        self.assertEqual(len(p._objects2(p._input(inp))(True)(True)), 8)


class TestOcamlType(unittest.TestCase):
    def test1(self):
        from dreamcoder.type import arrow, baseType, t0, tlist
        tgrid = baseType("grid") # any modified grid
        tbase_bool = baseType('base_bool')

        type1 = arrow(tgrid, tgrid)
        expected = "(tgrid @> tgrid)"
        self.assertEqual(type1.ocaml_type(), expected)

        type1 = tgrid
        expected = '(tgrid)'
        self.assertEqual(type1.ocaml_type(), expected)

        type1 = arrow(tgrid, t0)
        expected = '(tgrid @> t0)'
        self.assertEqual(type1.ocaml_type(), expected)

        type1 = arrow(tgrid, tlist(t0))
        expected = '(tgrid @> tlist(t0))'
        self.assertEqual(type1.ocaml_type(), expected)

        type1 = arrow(tgrid, tgrid, tgrid, tgrid)
        expected = "(tgrid @> tgrid @> tgrid @> tgrid)"
        self.assertEqual(type1.ocaml_type(), expected)

        type1 = arrow(arrow(tgrid, tgrid, tgrid), tbase_bool, tbase_bool, tgrid)
        expected = '((tgrid @> tgrid @> tgrid) @> tbase_bool @> tbase_bool @> tgrid)'
        self.assertEqual(type1.ocaml_type(), expected)

    def test2(self):
        from dreamcoder.type import arrow, baseType
        tgrid = baseType("tgrid") # any modified grid
        tbase_bool = baseType('tbase_bool')

        type1 = tgrid
        self.assertEqual(type1.number_of_args(), 0)

        type1 = arrow(tgrid, tgrid)
        self.assertEqual(type1.number_of_args(), 1)

        type1 = arrow(tgrid, tgrid, tgrid, tgrid)
        self.assertEqual(type1.number_of_args(), 3)

        type1 = arrow(arrow(tgrid, tgrid, tgrid), tbase_bool, tbase_bool, tgrid)
        self.assertEqual(type1.number_of_args(), 3)

    def test3(self):
        self.maxDiff = None

        a = pd['color_in']
        out = 'let primitive_color_in = primitive "color_in" (tgrid @> tcolor @> tgrid) (fun x y -> x);;'
        self.assertEqual(a.ocaml_string(), out)

        a = pd['map_i_to_j']
        out = 'let primitive_map_i_to_j = primitive "map_i_to_j" (tgrid @> tcolor @> tcolor @> tgrid) (fun x y z -> x);;'
        self.assertEqual(a.ocaml_string(), out)

        a = pd['T']
        out = 'let primitive_T = primitive "T" (tbase_bool) (0);;'
        self.assertEqual(a.ocaml_string(), out)

        a = pd['construct_mapping']
        out = 'let primitive_construct_mapping = primitive "construct_mapping" ((tgrid @> tlist(tgrid)) @> (tgrid @> tlist(tgrid)) @> tinvariant @> tinput @> tlist(tgrid)) (fun x y z w -> x);;'
        self.assertEqual(a.ocaml_string(), out)
        
        a = pd['filter_list']
        out = 'let primitive_filter_list = primitive "filter_list" (tlist(t0) @> (t0 @> tbool) @> tlist(t0)) (fun x y -> x);;'
        self.assertEqual(a.ocaml_string(), out)





if __name__ == '__main__':
    unittest.main()

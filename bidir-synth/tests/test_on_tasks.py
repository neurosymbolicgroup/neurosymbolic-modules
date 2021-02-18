import unittest
from typing import Callable, Union

from bidir.task_utils import get_arc_task_examples
from bidir.primitives.types import Grid, Color
import bidir.primitives.functions as F

ArcProgram = Callable[[Grid], Grid]


class TestOnTasks(unittest.TestCase):
    def check_arc_train_task(
        self,
        task_num: int,
        program: ArcProgram,
    ) -> None:
        train_examples, test_examples = get_arc_task_examples(task_num,
                                                              train=True)
        grid_pairs = train_examples + test_examples
        for in_grid, out_grid in grid_pairs:
            pred_grid = program(in_grid)

            self.assertEqual(
                pred_grid,
                out_grid,
                msg=(f"\n"
                     f"task number: {task_num}\n"
                     f"in  : {in_grid}\n"
                     f"out : {out_grid}\n"
                     f"pred: {pred_grid}\n"),
            )

    def get_train_program(
        self,
        task_num: int,
    ) -> Union[ArcProgram, str]:
        # yapf: disable
        if task_num == 0:
            def solve(x):
                obj = F.set_bg(x, Color.BLACK)
                obj = F.kronecker(obj, obj)
                return F.unset_bg(obj, Color.BLACK)
            return solve
        elif task_num == 30:
            return lambda x: F.unset_bg(F.crop(F.set_bg(x, Color.BLACK)),
                                        Color.BLACK)
        elif task_num == 31:
            def solve(x):
                columns = F.columns(x)

                def vblock(height, color):
                    return F.block(height, 1, color)

                def f1(col):
                    col = F.set_bg(col, Color.BLACK)
                    return vblock(F.area(col), F.get_color(col))

                def f2(col):
                    col = F.filter_color(col, Color.BLACK)
                    return vblock(F.area(col), Color.BLACK)

                def f3(col):
                    return F.vstack_pair(f2(col), f1(col))

                blocks = F.map_fn(f3, columns)
                return F.hstack(blocks)
            return solve
        elif task_num == 38:
            def solve(x):
                obj = F.crop(F.set_bg(x, Color.BLACK))
                top_half = F.top_half(obj)
                top_left = F.rotate_ccw(F.top_half(F.rotate_cw(top_half)))
                return F.unset_bg(top_left, Color.BLACK)
            return solve
        elif task_num == 56:
            def solve2(x):
                obj = F.crop(F.set_bg(x, Color.BLACK))

                def mirror_diag(g):
                    return F.rotate_cw(F.vflip(g))

                out = mirror_diag(F.vstack_pair(mirror_diag(obj),
                                  mirror_diag(obj)))

                return F.unset_bg(out, Color.BLACK)

            def solve(x):
                obj = F.crop(F.set_bg(x, Color.BLACK))
                hblock = F.block(1, 2, Color.BLACK)
                out = F.kronecker(hblock, obj)
                return F.unset_bg(out, Color.BLACK)
            return solve2
        # elif task_num == 78:
        #     def solve(x):
        #         x = F.set_bg(x, Color.BLACK)
        #         print('x!!: {}'.format(x))
        #         objs = F.objects(x, connect_colors=False,
        #                 connect_diagonals=True)
        #         print('objs: {}'.format(objs))
        #         freqs = F.frequency(objs)
        #         sorted_objs = F.sort_by_key(objs, freqs)
        #         most_common_first = F.reverse(sorted_objs)
        #         obj = F.get(most_common_first, 0)
        #         return F.unset_bg(obj, Color.BLACK)
        #     return solve
        elif task_num == 82:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 86:
            return lambda x: F.rotate_cw(F.rotate_cw(x))
        elif task_num == 99:
            def solve(x):
                block = F.block(2, 2, F.get_color(F.set_bg(x, Color.BLACK)))
                return block
            return solve
        elif task_num == 105:
            def solve(x):
                x1 = F.rotate_cw(x)
                x2 = F.rotate_cw(x1)
                x3 = F.rotate_cw(x2)
                top = F.hstack_pair(x, x1)
                bottom = F.hstack_pair(x3, x2)
                return F.vstack_pair(top, bottom)
            return solve
        # elif task_num == 110:
        #     def solve(x):
        #         x = F.set_bg(x, Color.BLACK)
        #         objs = F.objects(x, connect_colors=True,
        #                 connect_diagonals=True)

        #         def f(o):
        #             return F.contains_color(o, Color.GREY)

        #         filtered_objs = F.filter_by_fn(fn=f, xs=objs)
        #         obj = F.get(filtered_objs, 0)
        #         obj = F.set_bg(obj, Color.GREY)
        #         obj = F.crop(obj)
        #         return F.unset_bg(obj, Color.BLACK)
        #     return solve
        elif task_num == 112:
            def solve(x):
                x = F.set_bg(x, Color.BLACK)
                x = F.overlay_pair(x, F.vflip(x))
                return F.unset_bg(x, Color.BLACK)
            return solve
        elif task_num == 115:
            return lambda x: F.vstack_pair(F.vflip(x), x)
        elif task_num == 128:
            return lambda x: F.color_in(x, F.get_color(x))
        elif task_num == 139:
            return lambda x: F.rotate_ccw(F.rotate_ccw(x))
        elif task_num == 141:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 149:
            return lambda x: F.hflip(x)
        elif task_num == 151:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 154:
            # solving more circuitously to test map function
            def solve(x):
                cols = F.columns(x)
                flipped_cols = F.map_fn(F.vflip, cols)
                out = F.hstack(flipped_cols)
                return out
            return solve
        elif task_num == 163:
            return lambda x: F.hstack_pair(x, F.hflip(x))
        elif task_num == 171:
            return lambda x: F.vstack_pair(x, F.vflip(x))
        # elif task_num == 173:
        # def solve(x):
        #     x = F.set_bg(x, Color.BLACK)
        #     objs = F.objects(x, connect_colors=True,
        #             connect_diagonals=True)
        #     objs = F.filter_by_fn(fn=lambda obj:
        #             F.has_horizontal_symmetry(obj), xs=objs)
        #     out = F.get(objs, 0)
        #     return F.unset_bg(out, Color.BLACK)
        # return solve
        elif task_num == 176:
            return lambda x: F.hflip(F.crop(F.set_bg(x, Color.BLACK)))
        elif task_num == 178:
            return lambda x: F.hflip(F.rotate_cw(x))
        elif task_num == 194:
            def solve(x):
                deflated = F.deflate(F.crop(F.set_bg(x, Color.BLACK)), 3)
                return F.unset_bg(F.kronecker(deflated, deflated),
                                  Color.BLACK)
            return solve
        elif task_num == 209:
            return lambda x: F.vstack_pair(x, F.vflip(x))
        elif task_num == 210:
            def solve(x):
                y = F.hstack_pair(F.hflip(x), x)
                z = F.vflip(y)
                return F.vstack_pair(F.vstack_pair(z, y), z)
            return solve
        elif task_num == 216:
            def solve(x):
                obj = F.set_bg(x, Color.BLACK)
                obj = F.crop(obj)
                obj = F.kronecker(obj, obj)
                return F.unset_bg(obj, Color.BLACK)
            return solve
        elif task_num == 222:
            return lambda x: F.inflate(x, 3)
        elif task_num == 228:
            def solve(x):
                color = F.get_color(x)
                just_color = F.filter_color(x, color)
                greyed_out = F.color_in(x, Color.GREY)
                out = F.overlay_pair(just_color, greyed_out)
                return out
            return solve
        elif task_num == 240:
            return lambda x: F.hflip(F.rotate_cw(x))
        elif task_num == 248:
            return lambda x: F.hstack_pair(x, x)
        elif task_num == 256:
            def solve(x):
                x = F.set_bg(x, Color.BLUE)
                top = F.top_half(x)
                bottom = F.vflip(F.top_half(F.vflip(x)))

                def left_half(x):
                    return F.rotate_ccw(F.top_half(F.rotate_cw(x)))

                def right_half(x):
                    return F.rotate_cw(F.top_half(F.rotate_ccw(x)))

                top_left = left_half(top)
                top_right = right_half(top)
                bottom_left = left_half(bottom)
                bottom_right = right_half(bottom)

                crop_tl = F.crop(top_left)
                crop_tr = F.crop(top_right)
                crop_bl = F.crop(bottom_left)
                crop_br = F.crop(bottom_right)

                crop_tl = F.set_bg(crop_tl, Color.BLACK)
                crop_tr = F.set_bg(crop_tr, Color.BLACK)
                crop_bl = F.set_bg(crop_bl, Color.BLACK)
                crop_br = F.set_bg(crop_br, Color.BLACK)

                out = F.overlay_pair(crop_tl, crop_tr)
                out = F.overlay_pair(out, crop_bl)
                out = F.overlay_pair(out, crop_br)

                out = F.unset_bg(out, Color.BLACK)
                return out

            return solve
        elif task_num == 258:
            return lambda x: F.unset_bg(F.crop(F.set_bg(x, Color.BLUE)),
                                        Color.BLACK)
        elif task_num == 268:
            def solve(x):
                out = F.inflate(x, F.area(F.set_bg(x, Color.BLACK)))
                return F.unset_bg(out, Color.BLACK)
            return solve
        elif task_num == 275:
            return lambda x: F.color_i_to_j(x, Color.PINK, Color.RED)
        elif task_num == 289:
            def solve(x):
                obj = F.crop(F.set_bg(x, Color.BLACK))
                color = F.get_color(obj)
                obj = F.set_bg(obj, color)
                color2 = F.get_color(obj)
                obj = F.color_i_to_j(obj, color2, color)
                obj = F.color_i_to_j(obj, Color.BACKGROUND_COLOR, color2)
                return obj
            return solve
        elif task_num == 299:
            def solve(x):
                x = F.set_bg(x, Color.BLACK)
                color = F.get_color(x)
                x = F.filter_color(x, color)
                return F.unset_bg(F.crop(x), Color.BLACK)
            return solve
        elif task_num == 303:
            def solve(x):
                filtered = F.filter_color(x, F.get_color(x))
                return F.unset_bg(F.kronecker(filtered, x), Color.BLACK)
            return solve
        elif task_num == 306:
            return lambda x: F.inflate(x, 2)
        elif task_num == 308:
            return lambda x: F.color_i_to_j(x, Color.ORANGE, Color.GREY)
        elif task_num == 310:
            return lambda x: F.hstack_pair(x, F.hflip(x))
        elif task_num == 336:
            def solve(x):
                x = F.color_i_to_j(x, Color.CYAN, Color.BLACK)
                x = F.color_i_to_j(x, Color.GREY, Color.CYAN)
                x = F.color_i_to_j(x, Color.BLACK, Color.GREY)
                return x
            return solve
        elif task_num == 338:
            def solve(x):
                x = F.set_bg(x, Color.BLACK)
                a = F.area(x)
                c = F.get_color(x)
                return F.block(1, a, c)
            return solve
        elif task_num == 359:
            def solve(x):
                left = F.rotate_ccw(F.top_half(F.rotate_cw(x)))
                left = F.crop(F.set_bg(left, Color.GREY))
                right = F.rotate_cw(F.top_half(F.rotate_ccw(x)))
                right = F.crop(F.set_bg(right, Color.GREY))
                left = F.set_bg(left, Color.BLACK)
                right = F.set_bg(right, Color.BLACK)
                out = F.overlay_pair(left, F.hflip(right))
                return F.unset_bg(out, Color.BLACK)
            return solve
        elif task_num == 379:
            return lambda x: F.rotate_ccw(x)
        elif task_num == 383:
            def solve(x):
                obj = F.inflate(F.crop(F.set_bg(x, Color.BLACK)), 2)
                return F.unset_bg(obj, Color.BLACK)
            return solve
        elif task_num == 384:
            def solve(x):
                x = F.set_bg(x, Color.BLACK)
                x = F.overlay_pair(x, F.vflip(x))
                return F.unset_bg(x, Color.BLACK)
            return solve
        elif task_num == 388:
            def solve(x):
                obj = F.set_bg(x, Color.GREY)
                color = F.get_color(obj)
                obj = F.color_i_to_j(obj, color, Color.BLACK)
                obj = F.color_i_to_j(obj, Color.BACKGROUND_COLOR, color)
                return obj
            return solve
        elif task_num == 288:
            def solve(x):
                temp = F.set_bg(x,Color.BLACK)
                i = F.length(F.colors(temp))
                obj = F.inflate(x,i)
                return obj
            return solve
        # elif task_num == 396:
        #     def solve(x):
        #         x = F.set_bg(x, Color.BLACK)
        #         objs = F.objects(x, connect_colors=True,
        #                 connect_diagonals=False)

        #         def f(o):
        #             return F.vstack_pair(o,
        #                     F.block(F.length(F.colors(o)), 2, Color.GREEN))

        #         objs = F.map_fn(f, objs)
        #         return F.place_into_grid(objs, x)
        #     return solve
        else:
            return "No program"
        # yapf: enable

    def test_on_train_tasks(self):
        total_solved = 0

        for task_num in range(400):
            program = self.get_train_program(task_num)
            if isinstance(program, str):
                continue

            with self.subTest(task_num=task_num):
                self.assertNotEqual(program, None,
                                    (f"program for {task_num} is None."
                                     f"Did you forget to 'return solve'?"))
                self.check_arc_train_task(task_num, program)
                total_solved += 1

        print(f"\nSolved {total_solved} ARC train tasks")

import binutil  # need for importing dreamcoder properly
import datetime
import os

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar

from dreamcoder.domains.arc.arcPrimitives import (
    primitive_dict,
    generate_ocaml_primitives,
)
from dreamcoder.domains.arc.main import ArcNet
from dreamcoder.domains.arc.makeTasks import get_arc_task
from dreamcoder.domains.arc.task_testing import check_tasks
from dreamcoder.domains.arc.test import test

COPY_1_PRIM_NAMES = [
    "objects2", "T", "F", "input", "rotation_invariant", "size_invariant",
    "color_invariant", "no_invariant", "place_into_grid", "rows", "columns",
    "construct_mapping", "vstack", "hstack"
]
COPY_2_PRIM_NAMES = [
    "objects2", "T", "F", "input", "rotation_invariant", "size_invariant",
    "color_invariant", "no_invariant", "construct_mapping2",
    "construct_mapping3", "area", "has_y_symmetry", "list_length",
    "filter_list", "contains_color", "color2"
]
INFLATE_PRIM_NAMES = [
    "input", "object", "area", "kronecker", "inflate", "deflate", "2", "3",
    "num_colors"
]
SYMMETRY_PRIM_NAMES = [
    "object", "x_mirror", "y_mirror", "rotate_cw", "rotate_ccw", "left_half",
    "right_half", "top_half", "bottom_half", "overlay",
    "combine_grids_vertically", "combine_grids_horizontally", "input"
]
MISC_PRIM_NAMES = [
    "map_i_to_j", "list_of_one", "place_into_grid", "place_into_input_grid",
    "sort_incr", "sort_decr", "color_in", "color", "overlay", "object",
    "objects2", "T", "F", "hblock", "vblock", "area", "input", "move_down2",
    "get_first", "shell", "hollow", "fill_rectangle", "enclose_with_ring",
    "is_rectangle", "is_rectangle_not_pixel"
]


def main():
    #prim_names = primitive_dict.keys()
    prim_names = (COPY_1_PRIM_NAMES + COPY_2_PRIM_NAMES + INFLATE_PRIM_NAMES +
                  SYMMETRY_PRIM_NAMES + MISC_PRIM_NAMES)

    # need to de-duplicate
    primitives = list(set(primitive_dict[pn] for pn in prim_names))

    if "ARC_TWANG_GEN_PRIMS" in os.environ:
        generate_ocaml_primitives(primitives)
        return

    tasks = [get_arc_task(i) for i in range(0, 400)]
    grammar = Grammar.uniform(primitives)

    args = commandlineArguments(
        # Compute budget
        iterations=5,
        enumerationTimeout=120,
        recognitionTimeout=3600,
        # Runtime config
        CPUs=15,
        cuda=False,
        solver="python",
        # Specifics
        featureExtractor=ArcNet,
        auxiliary=True,  # train our feature extractor too
        contextual=True,  # use bi-gram model, not unigram
        a=3,  # max arity of compressed primitives
        topK=2,
        maximumFrontier=5,  # number of programs used for compression
        aic=.1,  # LOWER THAN USUAL, to incentivize making primitives
        pseudoCounts=30.0,
        helmholtzRatio=0.5,
    )

    generator = ecIterator(
        grammar=grammar,
        tasks=tasks,
        testingTasks=[],
        outputPrefix=f"./experimentOutputs/arc/{str(datetime.date.today())}",
        **args,
    )

    for i, _ in enumerate(generator):
        print(f"Done with ecIterator iter #{i}.")


if __name__ == "__main__":
    main()

# TODO: penalty for repeating an action twice
"""
This file is where we define all of the reward values for the agent.
Putting them here so they're easy to access and change without searching all
over the code for them.
"""

"""Giving an op that directly leads to discovered task solution."""
SOLVE_TASK = 1
"""Taking max number of actions."""
TIME_OUT = -1
"""Solving the train but not test examples."""
SOLVE_TRAIN_BUT_NOT_TEST_EXAMPLES = -1

"""Op wasn't used in discovered program for task."""
OP_NOT_USED_IN_SOLUTION = -1
"""
Op was used in discovered program for task.
On one hand, since we used discounted future rewards to give the reward for an
earlier op, those discounted rewards might be enough to encourage this op in
the future.
On the other hand, given a program out = f1(f2(f3(f4(in)))), intuitively we
should reward each function equally, regardless of the order in which the
actions were applied.
"""
OP_USED_IN_SOLUTION = 1


"""Creating a cycle, e.g. vfip(vlip(in))."""
CREATE_CYCLE = -1
"""
Finding a shorter path to an already grounded value.
not sure if this should be positive, otherwise it might be beneficial to
create inefficient paths and then fix them.
"""
SHORTER_PATH_TO_GROUNDED_VAL = 0
"""
Making a longer path to an already grounded value.
Not sure if this is needed, since we penalize total time taken anyway.
If a shorter path is found, should retroactively apply to the longer path.
"""
LONGER_PATH_TO_GROUNDED_VAL = -0.5

# Invalid arg errors
"""When one of the types of the input/output args to a fn is incorrect."""
TYPE_ERROR = -1
"""When the number of args provided is incorrect."""
ARITY_ERROR = TYPE_ERROR
"""
This occurs whenever the arguments to a function/inverse fn/cond-inverse fn are
incorrect, even if the types are correct, e.g. providing two grids whose widths
aren't equal, or calling vstack_pair_cond_inv(out, None, None).
"""
SOFT_TYPE_ERROR = -1

import time
import math
import random
from typing import Tuple, List, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from modules.base_modules import FC
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph
import modules.synth_modules


from bidir.utils import assertEqual
from bidir.primitives.types import Grid, MIN_COLOR, NUM_COLORS
import rl.ops.twenty_four_ops
import rl.ops.arc_ops
from rl.random_programs import ActionSpec, random_24_program, ProgramSpec, random_arc_small_grid_inputs_sampler, random_bidir_program

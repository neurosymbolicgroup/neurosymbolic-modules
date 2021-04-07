# simplest dreamcoder tutorial

## Setup
ssh
```
ssh openmind
```

go to ec directory
```
cd /om2/user/$USER/ec
```

load singularity
```
module load openmind/singularity
```

## Put primitives in python file

To write in `bin/addition.py`:

```
import datetime
import os
import random

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# create primitives

def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2

primitives = [
    # Primitive(name in Ocaml, type, name in Python)
    Primitive("incr", arrow(tint, tint), _incr),
    Primitive("incr2", arrow(tint, tint), _incr2),
]

# create grammar
grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=10, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())


# helper function that will 
#   add some number `N` to a pseudo-random number:
# The return value is a dictionary format 
#   we will use to store the inputs and the outputs for each task.
def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}

# Each task will consist of 3 things:
# 1. a name
# 2. a mapping from input to output type (e.g. `arrow(tint, tint)`)
# 3. a list of input-output pairs
def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )

# Training data
def add1(): return addN(1)
def add3(): return addN(3)
def add6(): return addN(6)

training_examples = [
    {"name": "add3", "examples": [add3() for _ in range(5000)]}
]

training = [get_tint_task(item) for item in training_examples]

# Testing data
testing_examples = [
    {"name": "add6", "examples": [add6() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))

```

## Put primitives in ocaml file
To write in `solvers/program.ml`:

```
let primitive_increment2 = primitive "incr2" (tint @> tint) (fun x -> 2+x);;
```

compile the ocaml
```
make
```


## Run dreamcoder

```
srun singularity exec container.img python bin/addition.py -t 2 --testingTimeout 2 -i 2
```

TO UNDERSTAND THE GENERAL PROGRAM OUTPUT: 
the code runs for 2 iterations. Each iteration contains the wake and sleep components.  during first iteration, it synthesizes the training task and the testing task.  during the second iteration, it re-synthesizes the training task and the testing task.  

TO UNDERSTAND THIS LINE OF THE PROGRAM OUTPUT: 
`Launching int -> int (3 tasks) w/ 48 CPUs. 0.000000 <= MDL < 1.500000. Timeout 2.000000.`
This is one “enumeration” step in dreamcoder.  Can appear in wake or dreaming phase.  
Means it’s searching for programs to solve the task (of intput type int, and output type int)
It checks whether the programs searched for solve any in the batch of 3 training tasks as it’s doing it (in this example above we only gave it one training task, but this is the output if you give it 3).  MDL is minimal description length — bounding the size of the programs solved.

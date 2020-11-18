import binutil
from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tstr, tscan_input
from dreamcoder.domains.scan.main import ScanNet
from dreamcoder.program import Primitive
from dreamcoder.type import arrow
from dreamcoder.grammar import Grammar
from dreamcoder.dreamcoder import commandlineArguments, ecIterator

training_data = import_data()
training_data = training_data

training_tasks = make_tasks(training_data)

def root_primitive(word):
    return Primitive('SCAN_' + word, tstr, word)

def root_primitive2(word):
    return Primitive('SCAN_' + word, arrow(tstr, tstr), lambda s: word if s == '' else word + ' ' + s)

concat = Primitive('SCAN_concat', arrow(tstr, tstr, tstr), lambda s1: lambda
        s2: s1 + ' ' + s2)

endl = Primitive('SCAN_endl', tstr, '');


# get words found in output.
outputs = [o.split(' ') for (i, o) in training_data]
words = set([w for out in outputs for w in out])
# we explicitly write them for ocaml, so make sure it's the same. Otherwise we
# need to add some more in the ocaml file
assert words == set(['LTURN', 'RTURN', 'WALK', 'JUMP', 'LOOK', 'RUN'])
# words = ['LTURN', 'RTURN', 'WALK', 'JUMP', 'LOOK', 'RUN']

primitives = [endl] + [root_primitive2(w) for w in words]

grammar = Grammar.uniform(primitives)

args = commandlineArguments(
    enumerationTimeout=6,
    iterations=5,
    recognitionTimeout=1,
    featureExtractor=ScanNet,
    auxiliary=True,
    helmholtzRatio=0.0,
    a=3,
    maximumFrontier=10,
    topK=2,
    pseudoCounts=30.0,
    solver='python',
    CPUs=1)

generator = ecIterator(grammar,
                       training_tasks,
                       testingTasks=[],
                       # outputPrefix='/experimentOutputs/arc/',
                       **args)

for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))

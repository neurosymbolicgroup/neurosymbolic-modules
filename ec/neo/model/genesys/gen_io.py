import numpy as np
from math import *

def generate_IO_examples(programs, N, L):
    """ Given a collection of programs, for each program randomly generates N IO examples,
        using the specified length L for the input arrays. """
    # An IO example is a 2-tuple (input_values, output_value)
    IO = [[] for program in programs]
    for pi, program in enumerate(programs):
        input_types = program.ins
        input_nargs = len(input_types)
        # Generate N input-output pairs
        for _ in xrange(N):
            input_value = [None]*input_nargs
            for a in xrange(input_nargs):
                minv, maxv = program.bounds[a]
                if input_types[a] == int:
                    input_value[a] = np.random.randint(minv, maxv)
                elif input_types[a] == [int]:
                    input_value[a] = list(np.random.randint(minv, maxv, size=L))
                else:
                    raise Exception("Unsupported input type " + input_types[a] + " for random input generation")
            output_value = program.fun(input_value)
            IO[pi].append((input_value, output_value))
    return IO

# HELPER FUNCTIONS
def type_to_string(t):
    if t == int:
        return 'int'
    if t == [int]:
        return '[int]'
    if t == bool:
        return 'bool'
    if t == [bool]:
        return '[bool]'
    raise ValueError('Type %s cannot be converted to string.' % t)

def scanl1(f, xs):
    if len(xs) > 0:
        r = xs[0]
        for i in xrange(len(xs)):
            if i > 0:
                r = f.fun(r, xs[i])
            yield r

def SQR_bounds(A, B):
    l = max(0, A)   # inclusive lower bound
    u = B - 1       # inclusive upper bound
    if l > u:
        return [(0, 0)]
    # now 0 <= l <= u
    # ceil(sqrt(l))
    # Assume that if anything is valid then 0 is valid
    return [(-int(sqrt(u)), ceil(sqrt(u+1)))]

def MUL_bounds(A, B):
    return SQR_bounds(0, min(-(A+1), B))

def scanl1_bounds(l, A, B, L):
    if l.src == '+' or l.src == '-':
        return [(A/L+1, B/L)]
    elif l.src == '*':
        return [(int((max(0, A)+1) ** (1.0 / L)), int((max(0, B)) ** (1.0 / L)))]
    elif l.src == 'MIN' or l.src == 'MAX':
        return [(A, B)]
    else:
        raise Exception('Unsupported SCANL1 lambda, cannot compute valid input bounds.')

#Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])
class Function:
    def __init__(self, src, sig, fun, bounds):
        self.src = src
        self.sig = sig
        self.fun = fun
        self.bounds = bounds

#Program = namedtuple('Program', ['src', 'ins', 'out', 'fun', 'bounds'])
class Program:
    def __init__(self, src, ins, out, fun, bounds):
        self.src = src
        self.ins = ins
        self.out = out
        self.fun = fun
        self.bounds = bounds

def get_language(V):
    Null = V
    lambdas = [
        Function('IDT',     (int, int),          lambda i: i,                                         lambda (A, B): [(A, B)]),
        Function('INC',     (int, int),          lambda i: i+1,                                       lambda (A, B): [(A, B-1)]),
        Function('DEC',     (int, int),          lambda i: i-1,                                       lambda (A, B): [(A+1, B)]),
        Function('SHL',     (int, int),          lambda i: i*2,                                       lambda (A, B): [((A+1)/2, B/2)]),
        Function('SHR',     (int, int),          lambda i: int(float(i)/2),                           lambda (A, B): [(2*A, 2*B)]),
        Function('doNEG',   (int, int),          lambda i: -i,                                        lambda (A, B): [(-B+1, -A+1)]),
        Function('MUL3',    (int, int),          lambda i: i*3,                                       lambda (A, B): [((A+2)/3, B/3)]),
        Function('DIV3',    (int, int),          lambda i: int(float(i)/3),                           lambda (A, B): [(A, B)]),
        Function('MUL4',    (int, int),          lambda i: i*4,                                       lambda (A, B): [((A+3)/4, B/4)]),
        Function('DIV4',    (int, int),          lambda i: int(float(i)/4),                           lambda (A, B): [(A, B)]),
        Function('SQR',     (int, int),          lambda i: i*i,                                       lambda (A, B): SQR_bounds(A, B)),
        Function('isPOS',   (int, bool),         lambda i: i > 0,                                     lambda (A, B): [(A, B)]),
        Function('isNEG',   (int, bool),         lambda i: i < 0,                                     lambda (A, B): [(A, B)]),
        Function('isODD',   (int, bool),         lambda i: i % 2 == 1,                                lambda (A, B): [(A, B)]),
        Function('isEVEN',  (int, bool),         lambda i: i % 2 == 0,                                lambda (A, B): [(A, B)]),
        Function('+',       (int, int, int),     lambda i, j: i+j,                                    lambda (A, B): [(A/2+1, B/2)]),
        Function('-',       (int, int, int),     lambda i, j: i-j,                                    lambda (A, B): [(A/2+1, B/2)]),
        Function('*',       (int, int, int),     lambda i, j: i*j,                                    lambda (A, B): MUL_bounds(A, B)),
        Function('MIN',     (int, int, int),     lambda i, j: min(i, j),                              lambda (A, B): [(A, B)]),
        Function('MAX',     (int, int, int),     lambda i, j: max(i, j),                              lambda (A, B): [(A, B)]),
    ]
    LINQ = [
        Function('REVERSE', ([int], [int]),      lambda xs: list(reversed(xs)),                       lambda (A, B, L): [(A, B)]),
        Function('SORT',    ([int], [int]),      lambda xs: sorted(xs),                               lambda (A, B, L): [(A, B)]),
        Function('TAKE',    (int, [int], [int]), lambda n, xs: xs[:n],                                lambda (A, B, L): [(0,L), (A, B)]),
        Function('DROP',    (int, [int], [int]), lambda n, xs: xs[n:],                                lambda (A, B, L): [(0,L), (A, B)]),
        Function('ACCESS',  (int, [int], int),   lambda n, xs: xs[n] if n>=0 and len(xs)>n else Null, lambda (A, B, L): [(0,L), (A, B)]),
        Function('HEAD',    ([int], int),        lambda xs: xs[0] if len(xs)>0 else Null,             lambda (A, B, L): [(A, B)]),
        Function('LAST',    ([int], int),        lambda xs: xs[-1] if len(xs)>0 else Null,            lambda (A, B, L): [(A, B)]),
        Function('MINIMUM', ([int], int),        lambda xs: min(xs) if len(xs)>0 else Null,           lambda (A, B, L): [(A, B)]),
        Function('MAXIMUM', ([int], int),        lambda xs: max(xs) if len(xs)>0 else Null,           lambda (A, B, L): [(A, B)]),
        Function('SUM',     ([int], int),        lambda xs: sum(xs),                                  lambda (A, B, L): [(A/L+1, B/L)]),
    ] + \
    [Function(
            'MAP ' + l.src,
            ([int], [int]),
            lambda xs, l=l: map(l.fun, xs),
            lambda (A, B, L), l=l: l.bounds((A, B))
        ) for l in lambdas if l.sig==(int, int)] + \
    [Function(
            'FILTER ' + l.src,
            ([int], [int]),
            lambda xs, l=l: filter(l.fun, xs),
            lambda (A, B, L), l=l: [(A, B)],
        ) for l in lambdas if l.sig==(int, bool)] + \
    [Function(
            'COUNT ' + l.src,
            ([int], int),
            lambda xs, l=l: len(filter(l.fun, xs)),
            lambda (A, B, L), l=l: [(-V, V)],
        ) for l in lambdas if l.sig==(int, bool)] + \
    [Function(
            'ZIPWITH ' + l.src,
            ([int], [int], [int]),
            lambda xs, ys, l=l: [l.fun(x, y) for (x, y) in zip(xs, ys)],
            lambda (A, B, L), l=l: l.bounds((A, B)) + l.bounds((A, B)),
        ) for l in lambdas if l.sig==(int, int, int)] + \
    [Function(
            'SCANL1 ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(scanl1(l, xs)),
            lambda (A, B, L), l=l: scanl1_bounds(l, A, B, L),
        ) for l in lambdas if l.sig==(int, int, int)]
    return LINQ, lambdas

def compiler(source_code, V, L, min_input_range_length=0):
    """ Taken in a program source code, the integer range V and the tape lengths L,
        and produces a Program.
        If L is None then input constraints are not computed.
        """
    # Source code parsing into intermediate representation
    LINQ, _ = get_language(V)
    LINQ_names = [l.src for l in LINQ]
    input_types = []
    types = []
    functions = []
    pointers = []
    for line in source_code.split('\n'):
        instruction = line[5:]
        if instruction in ['int', '[int]']:
            input_types.append(eval(instruction))
            types.append(eval(instruction))
            functions.append(None)
            pointers.append(None)
        else:
            split = instruction.split(' ')
            command = split[0]
            args = split[1:]
            # Handle lambda
            if len(split[1]) > 1 or split[1] < 'a' or split[1] > 'z':
                command += ' ' + split[1]
                args = split[2:]
            f = LINQ[LINQ_names.index(command)]
            assert len(f.sig) - 1 == len(args)
            ps = [ord(arg) - ord('a') for arg in args]
            types.append(f.sig[-1])
            functions.append(f)
            pointers.append(ps)
            assert [types[p] == t for p, t in zip(ps, f.sig)]
    input_length = len(input_types)
    program_length = len(types)
    # Validate program by propagating input constraints and check all registers are useful
    limits = [(-V, V)]*program_length
    if L is not None:
        for t in xrange(program_length-1, -1, -1):
            if t >= input_length:
                lim_l, lim_u = limits[t]
                new_lims = functions[t].bounds((lim_l, lim_u, L))
                num_args = len(functions[t].sig) - 1
                for a in xrange(num_args):
                    p = pointers[t][a]
                    limits[pointers[t][a]] = (max(limits[p][0], new_lims[a][0]),
                                              min(limits[p][1], new_lims[a][1]))
                    #print('t=%d: New limit for %d is %s' % (t, p, limits[pointers[t][a]]))
            elif min_input_range_length >= limits[t][1] - limits[t][0]:
                #print 'Program with no valid inputs: %s' % source_code
                return None
    # for t in xrange(input_length, program_length):
    #     print('%s (%s)' % (functions[t].src, ' '.join([chr(ord('a') + p) for p in pointers[t]])))
    # Construct executor
    my_input_types = list(input_types)
    my_types = list(types)
    my_functions = list(functions)
    my_pointers = list(pointers)
    my_program_length = program_length
    def program_executor(args):
        # print '--->'
        # for t in xrange(input_length, my_program_length):
        #     print('%s <- %s (%s)' % (chr(ord('a') + t), my_functions[t].src, ' '.join([chr(ord('a') + p) for p in my_pointers[t]])))
        assert len(args) == len(my_input_types)
        registers = [None]*my_program_length
        for t in xrange(len(args)):
            registers[t] = args[t]
        for t in xrange(len(args), my_program_length):
            registers[t] = my_functions[t].fun(*[registers[p] for p in my_pointers[t]])
        return registers[-1]
    return Program(
        source_code,
        input_types,
        types[-1],
        program_executor,
        limits[:input_length]
    )

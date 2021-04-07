from dreamcoder.task import Task
from dreamcoder.type import arrow, baseType

tstr = baseType('tstr')
tscan_input = baseType('tscan_input')

def import_data(path='data/SCAN/tasks.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        # remove \n
        lines = [l[:-1] for l in lines]
        # format: IN: jump  OUT: JUMP
        tasks = []
        for line in lines:
            # put into "normal" format
            line = line.replace('I_', '')
            line = line.replace('TURN_LEFT', 'LTURN')
            line = line.replace('TURN_RIGHT', 'RTURN')

            a = line.index('IN: ') + 4
            b = line.index('OUT: ') - 1
            b2 = b + 6
            i = line[a:b]
            o = line[b2:]
            tasks.append((i, o))

        tasks = sorted(tasks, key=lambda t: len(t[1]))


    return tasks


def make_tasks(scan_data):

    def make_task(input_str, output_str, name):
        # only one example per task
        examples = [((input_str,), output_str)]
        return Task(name, arrow(tscan_input, tstr), examples)

    return [make_task(i, o, str(n) + ': ' + i) for n, (i, o) in enumerate(scan_data)]




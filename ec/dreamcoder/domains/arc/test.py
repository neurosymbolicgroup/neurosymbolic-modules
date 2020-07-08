from dreamcoder.domains.arc.arcPrimitives import primitives
from dreamcoder.domains.arc.makeArcTasks import load_task
import numpy as np
import random

def test_recognition(ec_result):
    def new_transform(p, i1, i2, i3):
        import copy
        p = copy.deepcopy(p)
        p.body.x = i1
        p.body.f.x = i2
        p.body.f.f.x = i3
        return p

    model = ec_result.recognitionModel
    task, solution = list(ec_result.taskSolutions.items())[0]
    program = solution.entries[0].program
    print('program: {}'.format(program))
    input = task.examples[0][0][0]
    output = program.evaluate([])(input)
    print('(input, output): {}'.format((input, output)))

    grammar = model.grammarOfTask(task)
    grammar = grammar.untorch()
    score = grammar.logLikelihood(task.request, program)

    n = primitives[-10:]

    d = load_task(task_id)
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]


    
    for i, ex in enumerate(examples):
        results = []
        for i1 in range(10):
            for i2 in range(10):
                for i3 in range(10):
                    trio = i1, i2, i3
                    new_program = new_transform(program, n[i1], n[i2], n[i3])
                    output2 = new_program.evaluate([])(input)
                    score2 = grammar.logLikelihood(task.request, new_program)
                    results.append((trio, score2, new_program))


        results = sorted(results, key=lambda i: i[1])
        with open('results' + str(i) + '.out', 'w') as f:
            for r in results:
                f.write(str(r) + '\n')




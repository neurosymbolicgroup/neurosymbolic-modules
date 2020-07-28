import json
import os
import numpy as np

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

def run_stuff():
    d = load_task('0d3d703e')
    print(d)
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'])
    print(d['test'])

def find_example(grid, tasks):
    for d in tasks:
        present = np.any([np.array_equal(grid, i) for i in d["grids"]])
        if present:
            return d["name"]
    return None



def get_all_tasks():
    training_dir = 'data/ARC/data/training/'
    # take off last five chars of name to get rid of '.json'
    task_ids = [t[:-5] for t in os.listdir(training_dir)]

    def grids(task):
        grids = []
        for ex in task['train']:
            grids.append(np.array(ex['input']))
            grids.append(np.array(ex['output']))
        for ex in task['test']:
            grids.append(np.array(ex['input']))
            grids.append(np.array(ex['output']))

        return {"name": task["name"], "grids": grids}

    tasks = [load_task(task_id) for task_id in task_ids]
    tasks = [grids(task) for task in tasks]
    return tasks



if __name__ == '__main__':
    tasks = get_all_tasks()
    grid = np.array([[8, 5, 0], [8, 5, 3], [0, 3, 2]])
    print(find_example(grid, tasks))

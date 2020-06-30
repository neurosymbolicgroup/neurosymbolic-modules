import json
import os

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

def run_stuff():
    d = load_task('0d3d703e')
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'])
    print(d['test'])

if __name__ == '__main__':
    run_stuff()

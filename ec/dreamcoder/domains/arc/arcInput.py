import json
import os

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

if __name__ == '__main__':
    d = load_task('d07ae81c')
    print(d['train'][0]['input'])
    print(len(d['train'][0]['input'])) # height
    print(len(d['train'][0]['input'][0])) # width

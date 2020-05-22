from dreamcoder.domains.arc.arcInput import load_task

def get_task_dicts():
    return [load_task('0a938d79')]

def make_task(task_dict):
    task_name = task_dict['name']
    task_type = arrow(tgrid, tgrid)
    examples = format_examples(task_dict)
    return Task(task_name, task_type, examples)


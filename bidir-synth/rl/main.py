import numpy as np
import os
import random
import mlflow
from typing import Dict, Any, List, Tuple, Callable
from multiprocessing import Pool
import uuid

from bidir.task_utils import arc_task, twenty_four_task, Task
import bidir.utils as utils
from rl.agent import ManualAgent, RandomAgent, SynthAgent
from rl.environment import SynthEnv
import rl.ops.arc_ops
from rl.test_search import policy_rollouts
import rl.ops.twenty_four_ops
# from rl.ops.operations import ForwardOp
from rl.random_programs import (random_bidir_program,
                                # ProgramSpec,
                                random_arc_grid_inputs_sampler,
                                random_arc_small_grid_inputs_sampler,
                                random_arc_grid, random_program,
                                random_24_program)
from rl.policy_net import policy_net_24, policy_net_arc
from experiments.supervised_training import program_dataset, ActionDatasetOnDisk2
import experiments.supervised_training
import experiments.policy_gradient
import torch
# from ec.dreamcoder.program import Primitive

np.random.seed(3)
random.seed(3)


def run_until_done(agent: SynthAgent, env: SynthEnv):
    """
    Basic sketch of running an agent on the environment.
    There may be ways of doing this which uses the gym framework better.
    Seems like there should be a way of modularizing the choice of RL
    algorithm, the agent choice, and the environment choice.
    """

    done = False
    i = 0
    while not done:
        # for i, val in enumerate(env.psg.get_value_nodes()):
        #     ground_string = "G" if env.psg.is_grounded(val) else "UG"
        #     print(f'{i}:\t({ground_string}){type(val.value[0])})\t{str(val)}')

        action = agent.choose_action(env.observation())
        op, args = action
        state, reward, done, _ = env.step(action)

        # nodes = env.psg.get_value_nodes()
        # grounded_values = [g.value[0] for g in nodes
        #                    if env.psg.is_grounded(g)]
        # ungrounded_values = [
        #     g.value[0] for g in nodes if not env.psg.is_grounded(g)
        # ]
        # if reward >= 0:
        #     print(f"op: {op.name}, args: {args}")
        # s = input()
        # if i % 10 == 0:
        # print(f"{i} actions attempted")
        i += 1

    if env.is_solved():
        print(f"We solved the task in {i} actions")
        print(f"program: {env.psg.get_program()}")
    else:
        print("We timed out during solving the task.")


def arc_manual_agent():
    task = arc_task(288, train=True)
    env = SynthEnv(task=task, ops=rl.ops.arc_ops.ALL_OPS, max_actions=100)
    agent = ManualAgent(rl.ops.arc_ops.ALL_OPS)

    run_until_done(agent, env)


def twenty_four_manual_agent(numbers):
    task = twenty_four_task(numbers, 24)
    env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)
    agent = ManualAgent(rl.ops.twenty_four_ops.ALL_OPS)

    run_until_done(agent, env)


def twenty_four_random_agent(numbers):
    # num_actions = []
    # program_lengths = []
    # for _ in range(20000):
    # if _ % 1000 == 0:
    # print(_)
    task = twenty_four_task(numbers, 24)
    OPS = rl.ops.twenty_four_ops.ALL_OPS
    env = SynthEnv(task=task, ops=OPS, max_actions=1000)
    agent = RandomAgent(OPS)

    done = False
    i = 0
    while not done:
        action = agent.choose_action(env.observation())
        op, args = action
        state, reward, done, _ = env.step(action)
        i += 1

    print(f"number of actions: {i}")
    program_str = str(env.psg.get_program())
    print(f"program: {program_str}")
    # program_len = 1 + program_str.count('(')
    # num_actions.append(i)
    # program_lengths.append(program_len)

    # from matplotlib import pyplot as plt
    # plt.hist(num_actions, bins=50)
    # plt.xlabel('number of actions to find 24 program for inputs (2, 3, 4)')
    # plt.show()
    # plt.hist(program_lengths, bins=50)
    # plt.xlabel('length of 24 program discovered for inputs (2, 3, 4)')
    # plt.show()


def arc_random_agent():
    op_strs = ['block', '3', 'Color.RED', 'get_color']
    task_num = 128

    ops = [rl.ops.arc_ops.OP_DICT[s] for s in op_strs]

    success = 0
    for i in range(1, 2):
        # print('trying again')
        task = arc_task(task_num, train=True)
        env = SynthEnv(task=task, ops=ops, max_actions=100)
        agent = RandomAgent(ops)

        run_until_done(agent, env)
        prog = env.psg.get_program()
        succeeded = prog is not None
        success += succeeded
        if i % 10 == 0:
            print('success ratio: ' + str(success / i))


def peter_john_demo():
    # first, need to generate a local dataset. May take some time.
    parallel_arc_dataset_gen()
    # do supervised pretraining. Then fine-tune model on darpa tasks.
    arc_training()

    # to evaluate "test-time", need to modify the rollouts method to load the
    # model that was just training, and then call it.
    rollouts()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def arc_task_sampler(ops,
                     inputs_small: bool = False,
                     depth: int = 3):

    if inputs_small:
        inputs_sampler = random_arc_small_grid_inputs_sampler
    else:
        inputs_sampler = random_arc_grid_inputs_sampler

    def sampler():
        # return data.random_sample().task
        inputs = inputs_sampler()
        program_spec = random_bidir_program(ops=ops,
                                            inputs=inputs,
                                            depth=depth,
                                            forward_only=True)
        return program_spec.task

    return sampler


def arc_training():
    # Multithreading seems to just hog cpus without speeding up, so disable
    torch.set_num_threads(1)

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    # if None, loads a new model
    model_load_run_id = None
    model_load_run_id = "dd073931d6dd4ebb8ddbd81fffc47fd6"
    model_load_name = 'model'
    # model_load_name = 'epoch-1499'

    # run_supervised = True
    run_supervised = False
    # run_policy_gradient = False
    run_policy_gradient = True

    description = f"Check PG batch speed"

    forward_only = True
    if forward_only:
        data = forward_supervised_data()
    else:
        data = bidir_supervised_data()

    SUPERVISED_PARAMS = dict(
        save_model=True,
        save_every=500,
        epochs=2000,
        lr=0.002,  # default: 0.002
        print_every=100,
        use_cuda=use_cuda,
    )

    PG_TRAIN_PARAMS = dict(
        epochs=10000,
        max_actions=10,
        # TODO: test different batch sizes. 100, 500, 2000.
        batch_size=1000,
        # TODO: test different learning rates. 0.0001, 0.0005, 0.002
        lr=0.001,  # default: 0.001
        # TODO: maybe try reward_type='to-go',
        reward_type='shaped',
        save_model=True,
        save_every=200,
        forward_only=forward_only,
    )

    # this works for now.
    max_nodes = 20

    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    op_str = str(map(str, ops))
    op_str = op_str[0:min(249, len(op_str))]

    # saved by supervised too
    AUX_PARAMS: Dict[str, Any] = dict(
        description=description,
        model_load_run_id=model_load_run_id,
        ops=op_str,
        forward_only=forward_only,
        # for arc sampler
        arc_sampler_inputs_small=True,
        arc_sampler_depth=5,
    )

    darpa_tasks = arc_darpa_tasks()

    def darpa_sampler():
        return random.choice(darpa_tasks)

    arc_sampler = arc_task_sampler(ops,
            inputs_small = AUX_PARAMS['arc_sampler_inputs_small'],
            depth=AUX_PARAMS['arc_sampler_depth'])

    policy_gradient_sampler = arc_sampler

    if model_load_run_id:
        net = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    else:
        net = policy_net_arc(ops=ops, max_nodes=max_nodes, use_cuda=use_cuda)  # type: ignore
        print('Starting model from scratch')

    print(f"Number of parameters in model: {count_parameters(net)}")

    if run_supervised:
        mlflow.set_experiment("Supervised training 2")
        with mlflow.start_run():
            mlflow.log_params(SUPERVISED_PARAMS)
            mlflow.log_params(AUX_PARAMS)
            mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

            print(f"Starting run:\n{mlflow.active_run().info.run_id}")
            experiments.supervised_training.train(
                net,
                data,
                **SUPERVISED_PARAMS,  # type: ignore
            )

    if run_policy_gradient:
        # PG fine-tuning
        mlflow.set_experiment("Policy gradient 2")
        with mlflow.start_run():
            mlflow.log_params(PG_TRAIN_PARAMS)
            mlflow.log_params(AUX_PARAMS)
            mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

            print(f"Starting run:\n{mlflow.active_run().info.run_id}")
            experiments.policy_gradient.train(
                task_sampler=policy_gradient_sampler,
                ops=ops,
                policy_net=net,
                use_cuda=use_cuda,
                **PG_TRAIN_PARAMS,  # type: ignore
            )


def training_24():
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    random.seed(44)
    torch.manual_seed(44)

    # Empirically, I've found that torch's multithreading doesn't speed us up
    # at all, just hogs up cpus on polestar.. may be different for other
    # machines though
    torch.set_num_threads(1)
    data_size = 1000
    depth = 1
    num_inputs = 4
    max_input_int = 9
    max_int = rl.ops.twenty_four_ops.MAX_INT
    max_nodes = 6
    # model_load_run_id = "d903b68bd4dc4d808159da795d7ec866"  # depth-1 pretrained (76% acc)
    model_load_run_id = None
    model_load_name = 'epoch-2000'

    supervised_lr = 0.002  # default: 0.002

    save_model = True
    save_every_supervised = -1
    # save often in case we catastrophically forget..
    save_every_pg = 200
    supervised_epochs = 2
    # run_supervised = True
    run_supervised = False
    run_policy_gradient = True
    # run_policy_gradient = False
    description = """
    PG depth 3, loaded with 24 depth-1 supervised pretrained, then fine-tuned
    on depth-1.
    """

    # PG params
    TRAIN_PARAMS = dict(
        discount_factor=0.9,
        epochs=100,
        max_actions=10,
        # usually set to 1000
        batch_size=100,
        # default: 0.001
        lr=0.01,
        # mlflow can't log long lists
        # ops=OPS,
        reward_type='to-go',
        print_rewards_by_task=False,
        save_model=True,
        save_every=save_every_pg,
    )

    # saved by supervised too
    AUX_PARAMS: Dict[str, Any] = dict(
        description=description,
        model_load_run_id=model_load_run_id,
        depth=depth,
        num_inputs=num_inputs,
        max_input_int=max_input_int,
        enforce_unique=False,
        twenty_four=True,
    )

    # ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:5]
    # ops = rl.ops.twenty_four_ops.ALL_OPS
    ops = [rl.ops.twenty_four_ops.OP_DICT['add'],
           rl.ops.twenty_four_ops.OP_DICT['mul']]

    def sampler():
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
        return random_24_program(ops, inputs, depth)

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)

    def policy_gradient_sampler():
        return twenty_four_task((8, ), 16)

    if model_load_run_id:
        net = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    else:
        net = policy_net_24(ops, max_int=max_int, state_dim=512,
                            max_nodes=max_nodes,
                            use_cuda=use_cuda)
        print('starting model from scratch')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters in model: {count_parameters(net)}")

    if run_supervised:
        mlflow.set_experiment("Supervised training 24")
        with mlflow.start_run():
            PARAMS = dict(data_size=data_size, )

            mlflow.log_params(PARAMS)
            mlflow.log_params(AUX_PARAMS)
            mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

            print('Preview of data points:')
            for i in range(min(10, len(data))):
                # simply calling data[i] doesn't work
                print(data.__getitem__(i))

            print(f"Starting run:\n{mlflow.active_run().info.run_id}")
            experiments.supervised_training.train(
                net,
                data,
                epochs=supervised_epochs,
                print_every=1,
                lr=supervised_lr,
                save_model=save_model,
                save_every=save_every_supervised,
                use_cuda=use_cuda,
            )

    if run_policy_gradient:
        # PG fine-tuning
        mlflow.set_experiment("Policy gradient 24")

        # hopefully this starts a new run?
        with mlflow.start_run():

            mlflow.log_params(TRAIN_PARAMS)
            mlflow.log_params(AUX_PARAMS)
            mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

            experiments.policy_gradient.train(
                task_sampler=policy_gradient_sampler,
                ops=ops,
                policy_net=net,
                use_cuda=use_cuda,
                **TRAIN_PARAMS,  # type: ignore
            )


def arc_dataset(depth, num_samples=10000):
    '''
    Generates a dataset of random arc programs. This is done separately from
    training because generating samples is rather expensive, so it slows down
    the training a lot. This doesn't make 100% of sense to simon, because it
    seems like training the NN should be taking way longer, but for now,
    whatever.
    '''
    ops = rl.ops.arc_ops.GRID_OPS
    name = f"data/arc_depth{depth}"

    if not os.path.exists(name):
        os.mkdir(name)

    print(f'Generating examples of depth {depth}')

    while len(os.listdir(name)) < num_samples:
        # since we might generate multiple examples at a time
        if len(os.listdir(name)) % 1000 < depth:
            print(f'Reached {len(os.listdir(name))} examples..')
        inputs = [random_arc_grid()]

        program_spec = random_program(ops=ops, inputs=inputs, depth=depth)

        for action_spec in program_spec.action_specs:
            filename = name + '/' + str(uuid.uuid4())
            utils.save_action_spec(action_spec, filename)


def arc_darpa_tasks() -> List[Task]:
    task_nums = [
        82, 86, 105, 115, 139, 141, 149, 151, 154, 163, 171, 178, 209, 210,
        240, 248, 310, 379, 178
    ]
    tasks = [arc_task(task_num) for task_num in task_nums]
    return tasks


def hard_arc_darpa_tasks():
    hard_tasks = [arc_task(task_num) for task_num in [82, 105, 141, 151, 210]]
    hard_tasks2 = []
    for task in hard_tasks:

        def i_th_in_front(tupl):
            return tuple([tupl[i], *tupl])

        for i in range(len(task.target)):
            # make i'th example in front, so policy net sees it
            task2 = Task(tuple(i_th_in_front(inp) for inp in task.inputs),
                         i_th_in_front(task.target))
            hard_tasks2.append(task2)

    return hard_tasks2


def twenty_four_test_tasks():
    twenty_four_nums = [
        9551, 9521, 8851, 8842, 8754, 8664, 8643, 8622, 8621, 8521, 8432, 8411,
        7773, 7764, 7543, 7411, 6666, 6621, 6521, 6511, 6422, 5543, 4331, 4321,
        9887, 9884, 9883, 9854, 9761, 9741, 9654, 9642, 9632, 9611, 9532, 9441,
        8874, 8872, 8844, 8821, 8768, 8732, 8731, 8722, 8665, 8655, 8653, 8652,
        8531, 8441, 8422, 7754, 7741, 7732, 7642, 7621, 7542, 7541, 7531, 7443,
        7432, 7422, 7332, 7322, 6651, 6631, 6543, 6522, 6442, 6432, 5433, 5432,
        9885, 9862, 9844, 9832, 9743, 9742, 9622, 9421, 8875, 8852, 8831, 8744,
        8633, 8552, 8522, 7641, 7533, 7532, 7431, 7331, 6641, 5442, 5333, 5322
    ]
    twenty_four_inputs = [tuple(map(int, str(n))) for n in twenty_four_nums]
    twenty_four_tasks = [twenty_four_task(i, 24) for i in twenty_four_inputs]
    return twenty_four_tasks


def rollouts():
    torch.set_num_threads(1)

    model_load_run_id = "dd073931d6dd4ebb8ddbd81fffc47fd6"
    model_load_name = 'model'

    max_actions = 25
    model = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)

    # input is one, output is one
    # if you have more than one input, might need to increase.
    # I think it's okay to dynamically change the max nodes, since all empty
    # nodes are treated the same.
    model.max_nodes = max_actions + 2

    # make sure using same ops that the net was trained on!
    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    test_tasks = arc_darpa_tasks()
    # test_tasks = hard_arc_darpa_tasks()

    # note: if doing "for real", might want to change timeout time to do search
    # for much longer.
    timeout_seconds = 600

    solved_tasks, attempts_per_task = policy_rollouts(model=model,
                                                      ops=ops,
                                                      tasks=test_tasks,
                                                      timeout=timeout_seconds,
                                                      max_actions=max_actions)
    print(f"solved_tasks: {solved_tasks}")
    print(f"attempts_per_task: {attempts_per_task}")


def arc_bidir_dataset(path: str,
                      depth: int,
                      inputs_sampler: Callable[[], Tuple[Tuple[Any, ...],
                                                         ...]],
                      num_samples: int,
                      forward_only: bool = False):
    '''
    Generates a dataset of random arc programs. This is done separately from
    training because generating samples is rather expensive, so it slows down
    the training a lot. This doesn't make 100% of sense to simon, because it
    seems like training the NN should be taking way longer, but for now,
    whatever.

    This one is for bidirectional supervised examples.
    '''
    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    if not os.path.exists(path):
        os.mkdir(path)

    print(f'Generating examples of depth {depth}')

    while len(os.listdir(path)) < num_samples:
        # since each program gives multiple examples
        if len(os.listdir(path)) % 1000 < depth:
            print(f'Reached {len(os.listdir(path))} examples..')

        inputs = inputs_sampler()
        program_spec = random_bidir_program(ops=ops,
                                            inputs=inputs,
                                            depth=depth,
                                            forward_only=forward_only)
        for action_spec in program_spec.action_specs:
            # print(f"action_spec: {action_spec}")
            filepath = path + '/' + str(uuid.uuid4())
            utils.save_action_spec(action_spec, filepath)


def forward_supervised_data() -> ActionDatasetOnDisk2:
    return ActionDatasetOnDisk2(dirs=[
        'data/bidir/arc_fw_only_depth3',
        'data/bidir/arc_fw_only_depth5',
        'data/bidir/arc_fw_only_depth10',
        'data/bidir/arc_fw_only_depth20',
        'data/bidir/arc_fw_only_small_depth3',
        'data/bidir/arc_fw_only_small_depth5',
        'data/bidir/arc_fw_only_small_depth10',
        'data/bidir/arc_fw_only_small_depth20',
    ])


def bidir_supervised_data() -> ActionDatasetOnDisk2:
    return ActionDatasetOnDisk2(dirs=[
        'data/bidir/arc_bidir_depth3',
        'data/bidir/arc_bidir_depth5',
        'data/bidir/arc_bidir_depth10',
        'data/bidir/arc_bidir_depth20',
        'data/bidir/arc_bidir_small_depth3',
        'data/bidir/arc_bidir_small_depth5',
        'data/bidir/arc_bidir_small_depth10',
        'data/bidir/arc_bidir_small_depth20',
    ])


def arc_bidir_dataset_gen(args: Tuple):
    depth, forward_only, small = args
    if small:
        inputs_sampler = random_arc_small_grid_inputs_sampler
        if forward_only:
            path = f"data/bidir/arc_fw_only_small_depth{depth}"
        else:
            path = f"data/bidir/arc_bidir_small_depth{depth}"
    else:
        inputs_sampler = random_arc_grid_inputs_sampler
        if forward_only:
            path = f"data/bidir/arc_fw_only_depth{depth}"
        else:
            path = f"data/bidir/arc_bidir_depth{depth}"

    arc_bidir_dataset(path,
                      depth=depth,
                      inputs_sampler=inputs_sampler,
                      num_samples=10000,
                      forward_only=forward_only)


def parallel_arc_dataset_gen():
    args = [(depth, forward_only, small) for depth in [3, 5, 10, 20]
            for forward_only in [True, False] for small in [True, False]]

    with Pool() as p:
        p.map(arc_bidir_dataset_gen, args)
    # for arg in args:
        # arc_bidir_dataset_gen(arg)


def forward_vs_bidir_supervised():
    # generate dataset
    args = [(depth, forward_only, small) for depth in [3, 5, 10, 20]
            for forward_only in [True, False] for small in [True, False]]

    with Pool() as p:
        p.map(arc_bidir_dataset_gen, args)

    # do supervised training with both of them. compare accuracy
    # need to modify hyperparameters in here to fit
    arc_training()


if __name__ == '__main__':
    random.seed(45)
    torch.manual_seed(45)
    arc_training()

    # rollouts()
    # peter_john_demo()
    # parallel_arc_dataset_gen()
    # training_24()
    # hard_arc_darpa_tasks()

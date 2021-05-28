from typing import Any
import pickle
import os
import time
import mlflow
import torch.nn as nn

# hack for enabling/disabling cuda
USE_CUDA = False
# USE_CUDA = torch.cuda.is_available()


class SynthError(Exception):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """


def soft_assert(condition: bool):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """
    if not condition:
        raise SynthError


def assertEqual(a: Any, b: Any, message=''):
    assert a == b, f"{a} =/=  {b}. {message}"


def next_unused_path(path):
    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + f"__{i}" + extension
        i += 1

    return path


def save_mlflow_model(net: nn.Module, model_name='model'):
    # torch.save(net.state_dict(), save_path)
    # mlflow.pytorch.log_state_dict(net.state_dict(), save_path)
    mlflow.pytorch.log_model(net, model_name)
    print(f"Saved model for run\n{mlflow.active_run().info.run_id}",
          f"with name {model_name}")


def load_mlflow_model(run_id: str, model_name='model') -> nn.Module:
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    print(f"Loaded model from run {run_id} with name {model_name}")
    return model


def load_action_spec(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def repeat_n_times(sampler, n: int):
    """
    returns a new sampler which repeats each sample n times
    """
    repeated = 0
    sample = sampler()

    def new_sampler():
        nonlocal repeated
        nonlocal sample
        if repeated == n:
            sample = sampler()
            repeated = 0
        repeated += 1
        return sample

    return new_sampler


def save_action_spec(spec, path: str):
    with open(path, 'wb+') as f:
        pickle.dump(spec, f)


class timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            assert False, "Timing message should be string function"
        print("%s in %.1f seconds" % (message, dt))


def number_to_base(n, base=2):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % base))
        n //= base
    return digits[::-1]

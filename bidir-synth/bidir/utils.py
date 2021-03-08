from typing import Any
import os
import mlflow
import torch.nn as nn
import torch

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


def assertEqual(a: Any, b: Any):
    assert a == b, f"expected {b} but got {a}"


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



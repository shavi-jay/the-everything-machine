#!/usr/bin/env python
# coding: utf-8
"""
This script pretrains and finetunes MLPs on multiple sparse parity problems.

Comments
    -
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryHingeLoss

import wandb

from tqdm.auto import tqdm

import os
import inspect
import csv

from sparse_parity_v4_updated import (
    create_mlp,
    get_standard_multitask_batch,
    give_loss_fn,
    FastTensorDataLoader,
    cycle,
    save_model_weights,
    save_task,
)

# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------

alg_steps = 100000

config = dict(
    n_tasks=3,
    n=40,
    k=3,
    alpha=0.1,
    offset=0,
    D=-1,  # -1 for infinite data
    code_length=2,
    n_codes=3,
    width=100,
    depth=2,
    activation="ReLU",
    steps=alg_steps,  # 25000
    batch_size=128,
    lr=1e-4,
    weight_decay=0.05,
    test_points=10000,
    test_points_per_code=1000,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    log_freq=min(100, max(1, alg_steps // 1000)),
    verbose=False,
    loss_function_type="cross_entropy",
    is_early_stopping=True,
    early_stopping_threshold=5e-5,
    early_stopping_patience=100,
    save_model=True,
    save_model_path="fine_tune_models",
    save_task_path="fine_tune_codes",
    save_model_name="n_tasks_3_full_train_width_100_depth_5",
    pre_trained_model_load_path="models",
    pre_trained_model_load_file="n_tasks_3_full_train_width_100_depth_5",
    pre_trained_task_load_path="tasks",
    seed=0,
)


# --------------------------
#    ,-------------------.
#   (_\  HELPER FUNCTIONS \
#      |                  |
#      |                  |
#     _|                  |
#    (_/_____(*)_________/
#             \\
#              ))
#              ^
# --------------------------


def get_symmetric_difference_multitask_Ss(task_list, codes):
    """Creates Ss where control bits are not one-hot-encoded (combined parity of several tasks).

    Takes a symmetric difference of the task_list and codes to get Ss, the subsets of [1, ..., n] to compute sparse parities on.

    Parameters
    ----------
    task_list : list of lists of ints
        The tasks to combine.
    codes : list of lists of int
        The subtask indices which the batch will consist of.
    Returns
    -------
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    """
    Ss = []

    task_set = [set(task) for task in task_list]

    for code in codes:
        S = set()
        for subtask in code:
            S = S ^ task_set[subtask]
        Ss.append(sorted(list(S)))
    return Ss


def get_symmetric_difference_multitask_batch(
    n_tasks, n, task_list, codes, sizes, device="cpu", dtype=torch.float32
):
    """Creates batch for symmetric difference multitask problem, where we take symmetric difference of several tasks.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    task_list : list of lists of ints
        The tasks to combine.
    codes : list of int
        The subtask indices which the batch will consist of.
    sizes : list of int
        Number of samples for each subtask
    device : str
        Device to put batch on.
    dtype : torch.dtype
        Data type to use for input x. Output y is torch.int64.

    Returns
    -------
    x : torch.Tensor
        inputs
    y : torch.Tensor
        labels
    """

    Ss = get_symmetric_difference_multitask_Ss(task_list, codes)

    x, y = get_standard_multitask_batch(n_tasks, n, Ss, codes, sizes, device, dtype)

    return x, y


def load_pretrained_mlp(
    relative_path: str,
    save_file: str,
    activation: str,
    depth: int,
    width: int,
    n_tasks: int,
    n: int,
    device: str = "cpu",
):
    """Loads a pretrained MLP.

    Parameters
    ----------
    path: str
        Path to the pretrained model.
    activation: str
        Activation function to use.
    depth: int
        Number of layers.
    width: int
        Width of layers.
    n_tasks: int
        Number of tasks.
    n: int
        Bit string length for sparse parity problem.
    device: str
        Device to put batch on.

    Returns
    -------
    model: nn.Sequential
        The pretrained model.
    """

    model = create_mlp(activation, depth, width, n_tasks, n).to(device)

    dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    save_model_path = f"{relative_path}/{save_file}.pt"
    file_name = os.path.join(dirname, save_model_path)

    model.load_state_dict(torch.load(file_name))

    model.eval()

    return model

def load_task_list(relative_path: str, save_file: str):
    """Loads a task list from csv file.
    
    Parameters
    ----------
    relative_path: str
        Relative path to the task list.
    save_file: str
    
    Returns
    -------
    task_list: list of lists of int
        The task list.
    """

    dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    save_task_path = f"{relative_path}/{save_file}.csv"
    file_name = os.path.join(dirname, save_task_path)

    task_list = []
    with open(file_name, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            task_list.append(tuple([int(i) for i in row]))

    return task_list



def create_codes(n_codes: int, n_tasks: int, code_length: int, probs: list):
    """Creates codes for the sparse parity problem.

    Parameters
    ----------
    n: int
        Bit string length for sparse parity problem.
    k: int
        Number of 1s in the sparse parity problem.
    n_codes: int
        Number of codes to create.
    seed: int
        Seed for random number generator.

    Returns
    -------
    codes: list of lists of int
        The codes.
    codes_probs: list of float
        The empirical probabilities of each code.
    """

    codes = []
    code_probs = []

    for _ in range(n_codes * 10):
        code = np.random.choice(
            a=range(n_tasks), size=code_length, p=probs, replace=False
        )
        if sorted(list(code)) not in codes:
            codes.append(list(code))
            code_probs.append(np.prod(probs[code]))
        if len(codes) == n_codes:
            break

    assert (
        len(codes) == n_codes
    ), "Couldn't find enough codes for the given n_tasks, n_codes, code_length, alpha"

    code_probs = code_probs / np.sum(code_probs)

    return codes, code_probs


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------


def fine_tune_run(
    n_tasks: int,
    n: int,
    k: int,
    alpha: float,
    offset: int,
    D: int,
    code_length: int,
    n_codes: int,
    width: int,
    depth: int,
    activation: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    test_points: int,
    test_points_per_code: int,
    device: str,
    dtype: torch.dtype,
    log_freq: int,
    verbose: bool,
    loss_function_type: str,
    is_early_stopping: bool,
    early_stopping_threshold: float,
    early_stopping_patience: int,
    save_model: bool,
    save_model_path: str,
    save_task_path: str,
    save_model_name: str,
    pre_trained_model_load_path: str,
    pre_trained_model_load_file: str,
    pre_trained_task_load_path: str,
    seed: int,
):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    mlp: torch.nn.Sequential = load_pretrained_mlp(
        relative_path=pre_trained_model_load_path,
        save_file=pre_trained_model_load_file,
        activation=activation,
        depth=depth,
        width=width,
        n_tasks=n_tasks,
        n=n,
        device=device,
    )

    task_list = load_task_list(pre_trained_task_load_path, pre_trained_model_load_file)

    wandb.log({"task_list": task_list})

    probs = np.array(
        [np.power(n, -alpha) for n in range(1 + offset, n_tasks + offset + 1)]
    )
    probs = probs / np.sum(probs)

    wandb.log({"probs": probs})

    codes, code_probs = create_codes(n_codes, n_tasks, code_length, probs)

    wandb.log({"codes": codes})

    code_cdf = np.cumsum(code_probs)

    wandb.log({"normalised_code_probs": code_probs})

    test_batch_sizes = [int(code_prob * test_points) for code_prob in code_probs]

    wandb.log({"test_batch_sizes": test_batch_sizes})

    # Create finite batch
    if D != -1:
        samples = np.searchsorted(
            code_cdf,
            np.random.rand(
                D,
            ),
        )
        hist, _ = np.histogram(samples, bins=n_codes, range=(0, n_codes - 1))
        train_x, train_y = get_symmetric_difference_multitask_batch(
            n_tasks=n_tasks,
            n=n,
            task_list=task_list,
            codes=codes,
            sizes=hist,
            device=device,
            dtype=dtype,
        )
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(
            train_x, train_y, batch_size=min(D, batch_size), shuffle=True
        )
        train_iter = cycle(train_loader)

        wandb.log({"hist": hist})

    loss_fn = give_loss_fn(loss_function_type, device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    losses_subtasks = np.zeros(n_codes)
    accuracies_subtasks = np.zeros(n_codes)

    early_stopping_counter = 0

    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                test_x, test_y = get_symmetric_difference_multitask_batch(
                    n_tasks=n_tasks,
                    n=n,
                    task_list=task_list,
                    codes=codes,
                    sizes=test_batch_sizes,
                    device=device,
                    dtype=dtype,
                )
                test_y_pred = mlp(test_x)
                test_labels = torch.argmax(test_y_pred, dim=1)

                accuracies = torch.sum(test_labels == test_y).item() / test_points

                losses = loss_fn(test_y_pred, test_y).cpu().item()

                for i in range(n_codes):
                    x_i, y_i = get_symmetric_difference_multitask_batch(
                        n_tasks=n_tasks,
                        n=n,
                        task_list=task_list,
                        codes=[codes[i]],
                        sizes=[test_points_per_code],
                        device=device,
                        dtype=dtype,
                    )
                    y_pred_i = mlp(x_i)

                    labels_i_pred = torch.argmax(y_pred_i, dim=1)

                    losses_subtasks[i] = loss_fn(y_pred_i, y_i).cpu().item()

                    accuracies_subtasks[i] = (
                        torch.sum(labels_i_pred == y_i).item() / test_points_per_code
                    )

                log_step = step

                # find the squared norm of the parameters
                weight_norm_squared = 0
                for p in mlp.parameters():
                    weight_norm_squared += torch.sum(p**2).item()
                weight_norm = np.sqrt(weight_norm_squared)

                if is_early_stopping:
                    if accuracies >= 1 - early_stopping_threshold:
                        early_stopping_counter += 1
                    else:
                        early_stopping_counter = 0

                wandb.log(
                    {
                        "log_step": log_step,
                        "test_loss": losses,
                        "test_accuracy": accuracies,
                        "test_losses_subtasks": losses_subtasks,
                        "test_accuracies_subtasks": accuracies_subtasks,
                        "weight_norm": weight_norm,
                    }
                )

        optimizer.zero_grad()
        if D == -1:
            samples = np.searchsorted(
                code_cdf,
                np.random.rand(
                    batch_size,
                ),
            )
            hist, _ = np.histogram(samples, bins=n_codes, range=(0, n_codes - 1))

            train_x, train_y = get_symmetric_difference_multitask_batch(
                n_tasks=n_tasks,
                n=n,
                task_list=task_list,
                codes=codes,
                sizes=hist,
                device=device,
                dtype=dtype,
            )

        else:
    
            train_x, train_y = next(train_iter)

        train_y_pred = mlp(train_x)

        train_y_labels = torch.argmax(train_y_pred, dim=1)

        accuracies = torch.sum(train_y_labels == train_y).item() / batch_size

        loss = loss_fn(train_y_pred, train_y)

        if step % log_freq == 0:
            with torch.no_grad():
                train_loss = loss_fn(train_y_pred, train_y).item()
                wandb.log(
                    {
                        "training loss": train_loss,
                        "training accuracy": accuracies,
                        "y_hat prediction": wandb.Histogram(
                            torch.softmax(train_y_pred, dim=1)[:, 1].cpu().numpy()
                        ),
                        "odds ratio": wandb.Histogram(
                            train_y_pred[:, 1].cpu().numpy()
                            - train_y_pred[:, 0].cpu().numpy()
                        ),
                    }
                )

        loss.backward()
        optimizer.step()

        if is_early_stopping:
            if early_stopping_counter >= early_stopping_patience:
                break

    # save the model
    if save_model:
        save_model_weights(mlp, save_model_path, save_model_name)

        save_task(codes, save_task_path, save_model_name)

    return


if __name__ == "__main__":

    wandb.init(
    project="sparse_parity_grokking_finetuning",
    entity="shavi_jay",
    config=config,
    )

    fine_tune_run(**config)

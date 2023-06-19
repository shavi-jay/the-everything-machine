#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once.

Comments
    - now does sampling for everything except the test batch -- frequencies of subtasks are exactly distributed within test batch
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
alg_steps = 200000

config = dict(
    n_tasks=5,
    n=40,
    k=3,
    alpha=0.1,
    offset=0,
    D=-1,  # -1 for infinite data
    width=100,
    depth=2,
    activation="ReLU",
    steps=alg_steps,  # 25000
    batch_size=128,
    lr=1e-2,
    weight_decay=0.005,
    test_points=30000,
    test_points_per_task=1000,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    log_freq=min(100, max(1, alg_steps // 1000)),
    verbose=False,
    loss_function_type="cross_entropy",
    is_early_stopping=True,
    early_stopping_threshold=5e-5,
    early_stopping_patience=100,
    save_model=True,
    save_model_path="models",
    save_task_path="tasks",
    save_model_name="n_tasks_5",
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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(
                self.dataset_len, device=self.tensors[0].device
            )
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def get_batch(n_tasks, n, Ss, codes, sizes, device="cpu", dtype=torch.float32):
    """Creates batch.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of int
        The subtask indices which the batch will consist of
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
    batch_x = torch.zeros((sum(sizes), n_tasks + n), dtype=dtype, device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    start_i = 0
    for S, size, code in zip(Ss, sizes, codes):
        if size > 0:
            x = torch.randint(low=0, high=2, size=(size, n), dtype=dtype, device=device)
            y = torch.sum(x[:, S], dim=1) % 2
            x_task_code = torch.zeros((size, n_tasks), dtype=dtype, device=device)
            x_task_code[:, code] = 1
            x = torch.cat([x_task_code, x], dim=1)
            batch_x[start_i : start_i + size, :] = x
            batch_y[start_i : start_i + size] = y
            start_i += size
    return batch_x, batch_y


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def create_mlp(
    activation: str, depth: int, width: int, n_tasks: int, n: int, device: str = "cpu"
):
    if activation == "ReLU":
        activation_fn = nn.ReLU
    elif activation == "Tanh":
        activation_fn = nn.Tanh
    elif activation == "Sigmoid":
        activation_fn = nn.Sigmoid
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"

    # create model
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(n_tasks + n, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 2))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)

    return mlp


def cross_entropy_high_precision(y_pred, y_target):
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = torch.nn.functional.log_softmax(y_pred.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=y_target[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def give_loss_fn(loss_function_type: str, device: str):
    hinge_loss_fn = BinaryHingeLoss().to(device)
    hinge_loss_fn_logit = lambda y_hat, y: hinge_loss_fn(
        torch.softmax(y_hat, dim=1)[:, 1], y
    )

    if loss_function_type == "cross_entropy":
        loss_fn = lambda y_hat, y: cross_entropy_high_precision(y_hat, y)
    elif loss_function_type == "hinge":
        loss_fn = lambda y_hat, y: hinge_loss_fn_logit(y_hat, y)
    else:
        assert False, f"Unrecognized loss function identifier: {loss_function_type}"

    return loss_fn


def save_model_weights(model: torch.nn.Sequential, relative_path: str, save_file: str):
    # create relative path to save model
    dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    save_model_path = f"{relative_path}/{save_file}.pt"
    file_name = os.path.join(dirname, save_model_path)

    torch.save(model.state_dict(), file_name)

    return

def save_task(Ss: list, relative_path: str, save_file: str):
    with open (f'{relative_path}/{save_file}.csv','w',newline = '') as csvfile:
        my_writer = csv.writer(csvfile, delimiter = ' ')
        my_writer.writerows(Ss)

    return


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
def run(
    n_tasks: int,
    n: int,
    k: int,
    alpha: float,
    offset: int,
    D: int,
    width: int,
    depth: int,
    activation: str,
    test_points: int,
    test_points_per_task: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
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
    seed: int = 0,
):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    mlp: torch.nn.Sequential = create_mlp(activation, depth, width, n_tasks, n, device)

    Ss = []
    for _ in range(n_tasks * 10):
        S = tuple(sorted(list(random.sample(range(n), k))))
        if S not in Ss:
            Ss.append(S)
        if len(Ss) == n_tasks:
            break
    assert (
        len(Ss) == n_tasks
    ), "Couldn't find enough subsets for tasks for the given n, k"

    wandb.log({"Ss": Ss})

    probs = np.array(
        [np.power(n, -alpha) for n in range(1 + offset, n_tasks + offset + 1)]
    )
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    wandb.log({"probs": probs})

    test_batch_sizes = [int(prob * test_points) for prob in probs]

    if D != -1:
        samples = np.searchsorted(
            cdf,
            np.random.rand(
                D,
            ),
        )
        hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks - 1))
        train_x, train_y = get_batch(
            n_tasks=n_tasks,
            n=n,
            Ss=Ss,
            codes=list(range(n_tasks)),
            sizes=hist,
            device="cpu",
            dtype=dtype,
        )
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(
            train_x, train_y, batch_size=min(D, batch_size), shuffle=True
        )
        train_iter = cycle(train_loader)

        wandb.log({"hist": wandb.Histogram(hist)})

    loss_fn = give_loss_fn(loss_function_type, device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    losses_subtasks = np.zeros(n_tasks)
    accuracies_subtasks = np.zeros(n_tasks)

    early_stopping_counter = 0

    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                x_i, y_i = get_batch(
                    n_tasks=n_tasks,
                    n=n,
                    Ss=Ss,
                    codes=list(range(n_tasks)),
                    sizes=test_batch_sizes,
                    device=device,
                    dtype=dtype,
                )
                y_i_pred = mlp(x_i)
                labels_i_pred = torch.argmax(y_i_pred, dim=1)

                accuracies = torch.sum(labels_i_pred == y_i).item() / test_points

                losses = loss_fn(y_i_pred, y_i).item()

                for i in range(n_tasks):
                    x_i, y_i = get_batch(
                        n_tasks=n_tasks,
                        n=n,
                        Ss=[Ss[i]],
                        codes=[i],
                        sizes=[test_points_per_task],
                        device=device,
                        dtype=dtype,
                    )
                    y_i_pred = mlp(x_i)

                    labels_i_pred = torch.argmax(y_i_pred, dim=1)

                    losses_subtasks[i] = loss_fn(y_i_pred, y_i).item()

                    accuracies_subtasks[i] = (
                        torch.sum(labels_i_pred == y_i).item() / test_points_per_task
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
                cdf,
                np.random.rand(
                    batch_size,
                ),
            )
            hist, _ = np.histogram(samples, bins=n_tasks, range=(0, n_tasks - 1))
            x, y_target = get_batch(
                n_tasks=n_tasks,
                n=n,
                Ss=Ss,
                codes=list(range(n_tasks)),
                sizes=hist,
                device=device,
                dtype=dtype,
            )
        else:
            x, y_target = next(train_iter)

        y_pred = mlp(x)

        labels_pred = torch.argmax(y_pred, dim=1)

        accuracies = torch.sum(labels_pred == y_target).item() / y_target.shape[0]

        loss = loss_fn(y_pred, y_target)

        if step % log_freq == 0:
            with torch.no_grad():
                train_loss = loss_fn(y_pred, y_target).item()
                wandb.log(
                    {
                        "training loss": train_loss,
                        "training accuracy": accuracies,
                        "y_hat prediction": wandb.Histogram(
                            torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
                        ),
                        "odds ratio": wandb.Histogram(
                            y_pred[:, 1].cpu().numpy() - y_pred[:, 0].cpu().numpy()
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

        save_task(Ss, save_task_path, save_model_name)

    return


if __name__ == "__main__":

    wandb.init(
    project="sparse_parity_grokking",
    entity="shavi_jay",
    config=config,
    )

    run(**config)

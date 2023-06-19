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
import sys
import csv

import matplotlib.pyplot as plt
import matplotlib

from sparse_parity_v4_updated import create_mlp, get_batch, give_loss_fn

config = dict(
    n_tasks=3,
    n=40,
    width=100,
    depth=2,
    activation="ReLU",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
    model_name="test_3",
)


def get_Ss(model_name):
    Ss = []

    with open(f"tasks/{model_name}.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            Ss.append([int(element) for element in row])

    return Ss


def test_accuracy(Ss: list, mlp: nn.Sequential, loss_function_type: str, device: str, test_points_per_task: int):
    x_test, y_test = get_batch(
        config["n_tasks"],
        config["n"],
        Ss,
        list(range(config["n_tasks"])),
        sizes=[test_points_per_task] * config["n_tasks"],
        device=config["device"],
        dtype=config["dtype"],
    )

    y_pred = mlp(x_test)

    loss_fn = give_loss_fn(loss_function_type, device)

    loss = loss_fn(y_pred, y_test)

    # get accuracy

    labels = torch.argmax(y_pred, dim=1)

    test_accuracy = torch.sum(labels == y_test).item() / y_test.shape[0]

    print(f'Loss: {loss}; Test accuracy: {test_accuracy}')

    return test_accuracy


# create mlp model
mlp = create_mlp(
    activation=config["activation"],
    depth=config["depth"],
    width=config["width"],
    n_tasks=config["n_tasks"],
    n=config["n"],
    device=config["device"],
)

# load mlp model

mlp.load_state_dict(torch.load(f"models/{config['model_name']}.pt"))
mlp.eval()

# get Ss
Ss = get_Ss(config["model_name"])

test_accuracy(Ss, mlp, "cross_entropy", config["device"], 100)

print(Ss)

def plot_weights(mlp):

    with torch.no_grad():

        for i, (name, param) in enumerate(mlp.named_parameters()):
            print(name, param.data.shape)

            fig, ax = plt.subplots()


            if len(param.data.shape) != 2:

                plot_data = param.data.cpu().unsqueeze(1).numpy()

            else:
                plot_data = param.data.cpu().numpy()

            if param.data.shape[0] != 100:
                im = plt.imshow(plot_data)

                print(plot_data.shape)
            else:
                if plot_data.shape[1] == 100:
                    im = plt.imshow(plot_data.T)
                    print(plot_data.shape)
                else:
                    im = plt.imshow(plot_data.T)
                    print(plot_data.shape)
            plt.colorbar(im)

            im.set_cmap("bwr")

            # else:

            #     plt.bar(np.arange(param.data.shape[0]),param.data.cpu().numpy())
            plt.xlabel("Neuron index")

            plt.ylabel("Bit value")

            plt.title(f"Weight matrix for {name}")

            plt.show()

def compute_correlation(mlp: nn.Sequential):
    with torch.no_grad():
        for i, param in enumerate(mlp.parameters()):
            if i == 0:
                W = param.data.cpu().numpy()

                correlation_matrix = np.corrcoef(np.abs(W).T)

                fig, ax = plt.subplots()
                plt.imshow(correlation_matrix)

                plt.colorbar()

                plt.title("Correlation matrix for magnitude of first layer weights")

                plt.show()

def find_relevant_neurons(mlp: nn.Sequential, Ss: list, threshold: float, n_tasks: int):
    with torch.no_grad():
        for i, param in enumerate(mlp.parameters()):
            if i == 0:
                W = param.data.cpu().numpy()

                abs_W = np.abs(W)

                list_of_relevant_neurons = []

                for task in Ss:
                    weight_product = np.product(abs_W[:, np.array(task) + n_tasks], axis=1)

                    relevant_neurons = np.where(weight_product > threshold)[0]

                    list_of_relevant_neurons.append(relevant_neurons)

                    print(f"Task {task} has relevant neurons {relevant_neurons}")

                return list_of_relevant_neurons

list_of_relevant_neurons = find_relevant_neurons(mlp, Ss, 1, config["n_tasks"])

def weights_for_relevant_neurons(mlp: nn.Sequential, Ss, list_of_relevant_neurons, n_tasks):

    relevant_W_1_list = []
    relevant_b_1_list = []
    relevant_W_2_list = []
    relevant_b_2_list = []

    with torch.no_grad():
        for j, (task, relevant_neurons) in enumerate(zip(Ss, list_of_relevant_neurons)):

            print(f"Task {j + 1}: {task}")

            indices = np.append(np.arange(n_tasks),[np.array(task) + n_tasks])

            for i, (name, param) in enumerate(mlp.named_parameters()):

                
                if i == 0:
                    W = param.data.cpu().numpy()

                    relevant_W_1 = W[relevant_neurons][:, indices]

                    relevant_W_1_list.append(relevant_W_1)

                    print(f"Layer {name} has weights: \n {relevant_W_1}")
                if i == 1:
                    b = param.data.cpu().numpy()

                    relevant_b_1 = b[relevant_neurons]

                    relevant_b_1_list.append(relevant_b_1)

                    print(f"Layer {name} has biases: \n {relevant_b_1}")
                if i == 2:
                    W = param.data.cpu().numpy()

                    relevant_W_2 = W[:, relevant_neurons].T

                    relevant_W_2_list.append(relevant_W_2)

                    print(f"Layer {name} has weights: \n {relevant_W_2}")
                if i == 3:
                    b = param.data.cpu().numpy()

                    relevant_b_2 = b

                    relevant_b_2_list.append(relevant_b_2)

                    print(f"Layer {name} has biases: \n {relevant_b_2}")
    
    return relevant_W_1_list, relevant_b_1_list, relevant_W_2_list, relevant_b_2_list

relevant_W1_list, relevant_b1_list, relevant_W2_list, relevant_b2_list = weights_for_relevant_neurons(mlp, Ss, list_of_relevant_neurons, config["n_tasks"]) 

def compute_pre_activations(relevant_W1_list, relevant_b1_list, relevant_W2_list, relevant_b2_list, n_tasks, n):

    pre_activations_list = []

    for j, (relevant_W_1, relevant_b_1, relevant_W_2, relevant_b_2) in enumerate(zip(relevant_W1_list, relevant_b1_list, relevant_W2_list, relevant_b2_list)):

        per_task_pre_activations_list = []

        print(f"Task {j + 1} pre-activations")

        for i in range(n):
            x_test = np.zeros((n_tasks + n, 1))

            x_test[i + n_tasks, 0] = 1

            x_test[j, 0] = 1

            pre_activations = (relevant_W_1 @ x_test).T.squeeze() + relevant_b_1

            per_task_pre_activations_list.append(pre_activations)

            print(f"{pre_activations}")

        pre_activations_list.append(per_task_pre_activations_list)

    return pre_activations_list

pre_activations_list = compute_pre_activations(relevant_W1_list, relevant_b1_list, relevant_W2_list, relevant_b2_list, config["n_tasks"], config["n_tasks"])

plot_weights(mlp)

compute_correlation(mlp)



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

import copy

from sparse_parity_v4_updated import create_mlp, get_standard_multitask_batch, give_loss_fn

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
    x_test, y_test = get_standard_multitask_batch(
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

def find_relevant_neurons(mlp: nn.Sequential, Ss: list, threshold: float, n_tasks: int, print_weights: bool=False):
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

                    if print_weights:
                        print(f"Task {task} has relevant neurons {relevant_neurons}")

                return list_of_relevant_neurons

list_of_relevant_neurons = find_relevant_neurons(mlp, Ss, 1, config["n_tasks"])

def weights_for_relevant_neurons(mlp: nn.Sequential, Ss, list_of_relevant_neurons, n_tasks, print_weights=False):

    relevant_W_1_list = []
    relevant_b_1_list = []
    relevant_W_2_list = []
    relevant_b_2_list = []

    with torch.no_grad():
        for j, (task, relevant_neurons) in enumerate(zip(Ss, list_of_relevant_neurons)):
            if print_weights:
                print(f"Task {j + 1}: {task}")

            indices = np.append(np.arange(n_tasks),[np.array(task) + n_tasks])

            for i, (name, param) in enumerate(mlp.named_parameters()):

                
                if i == 0:
                    W = param.data.cpu().numpy()

                    relevant_W_1 = W[relevant_neurons][:, indices]

                    relevant_W_1_list.append(relevant_W_1)

                    if print_weights:
                        print(f"Layer {name} has weights: \n {relevant_W_1}")
                if i == 1:
                    b = param.data.cpu().numpy()

                    relevant_b_1 = b[relevant_neurons]

                    relevant_b_1_list.append(relevant_b_1)

                    if print_weights: 
                        print(f"Layer {name} has biases: \n {relevant_b_1}")
                if i == 2:
                    W = param.data.cpu().numpy()

                    relevant_W_2 = W[:, relevant_neurons].T

                    relevant_W_2_list.append(relevant_W_2)

                    if print_weights:
                        print(f"Layer {name} has weights: \n {relevant_W_2}")
                if i == 3:
                    b = param.data.cpu().numpy()

                    relevant_b_2 = b

                    relevant_b_2_list.append(relevant_b_2)

                    if print_weights:
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

#pre_activations_list = compute_pre_activations(relevant_W1_list, relevant_b1_list, relevant_W2_list, relevant_b2_list, config["n_tasks"], config["n_tasks"])

############################################

def set_non_relevant_weights_and_biases_to_zero(model: nn.Sequential, list_of_relevant_neurons: list, n_tasks: int, n: int, Ss: list):

    new_mlp = copy.deepcopy(model)

    with torch.no_grad():
        for i, param in enumerate(new_mlp.parameters()):
            if i == 0:
                W = param.data.cpu().numpy()

                print(W.shape)

                combined_list_of_relevant_neurons = []

                [combined_list_of_relevant_neurons.extend(relevant_neurons) for relevant_neurons in list_of_relevant_neurons]

                for k in range(W.shape[0]):
                    if k not in combined_list_of_relevant_neurons:
                        W[k] = np.zeros(W[k].shape)

                combined_list_of_relevant_weights = list(np.arange(n_tasks))

                [combined_list_of_relevant_weights.extend(np.array(task) + n_tasks) for task in Ss]

                for k in range(W.shape[1]):
                    if k not in combined_list_of_relevant_weights:
                        W[:,k] = np.zeros(W[:,k].shape)

                for j, (task, relevant_neurons) in enumerate(zip(Ss, list_of_relevant_neurons)):

                    indices = np.array(task) + n_tasks

                    print(f"Task {j + 1}: {task} has relevant neurons {relevant_neurons} and indices {indices}")

                    for k in range(W.shape[0]):
                        if k not in relevant_neurons:
                            W[k,:][indices] = np.zeros(W[k,:][indices].shape)

                param.data = torch.from_numpy(W).float().cuda()

            if i == 1:
                b = param.data.cpu().numpy()

                for k in range(b.shape[0]):
                    if k not in combined_list_of_relevant_neurons:
                        b[k] = np.zeros(b[k].shape)

                param.data = torch.from_numpy(b).float().cuda()
    return new_mlp

def round_weights_and_biases(model: nn.Sequential, decimal_place: int):
    
    new_mlp = copy.deepcopy(model)

    with torch.no_grad():
        for i, param in enumerate(new_mlp.parameters()):
            W = param.data.cpu().numpy()

            W = np.round(W, decimal_place)

            param.data = torch.from_numpy(W).float().cuda()

    return new_mlp

#Validating that when irrelevant weights and biases are set to zero, the accuracy is the same as the original MLP

new_mlp = set_non_relevant_weights_and_biases_to_zero(mlp, list_of_relevant_neurons, config["n_tasks"], config["n"], Ss)

mlp_accuracy = test_accuracy(Ss, mlp, "cross_entropy", config["device"], 100)
zeroed_mlp_accuracy = test_accuracy(Ss, new_mlp, "cross_entropy", config["device"], 100)
rounded_mlp_accuracy = test_accuracy(Ss, round_weights_and_biases(new_mlp, 0), "cross_entropy", config["device"], 1000)

print(f"Original MLP accuracy: {mlp_accuracy}")
print(f"Zeroed MLP accuracy: {zeroed_mlp_accuracy}")
print(f"Rounded MLP accuracy: {rounded_mlp_accuracy}")

#plot_weights(mlp)

#compute_correlation(mlp)



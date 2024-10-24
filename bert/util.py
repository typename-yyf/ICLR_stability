import os
import io
import json
from typing import Union
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torchvision

class Config():
    def __init__(self, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        for key, value in config_dict.items():
            setattr(self, key, value)

def lambdas(ntk: torch.Tensor):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    return eigenvalues

def singlevalues(matrix: torch.Tensor):
    singlevalues = torch.linalg.svdvals(matrix)
    return singlevalues

normalize = lambda t: (t - t.min()) / (t.max() - t.min())

def cal_rank(matrix: torch.Tensor):
    singlevalues = torch.linalg.svdvals(matrix)
    singlevalues_sorted, _ = torch.sort(singlevalues, descending=True)
    singlevalues_sorted_cumsum = torch.cumsum(singlevalues_sorted, dim=0)
    singlevalues_sorted_cumsum_norm = normalize(singlevalues_sorted_cumsum) 
    return singlevalues_sorted_cumsum_norm

def condition(ntk: torch.Tensor):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    lmin = eigenvalues[len(eigenvalues) // 2]
    lmax = eigenvalues[-1]
    condition_number = torch.nan_to_num(lmax / lmin, nan=1000000.0)
    return condition_number, lmax, lmin

def condition_norm(ntk: torch.Tensor):
    ntk = ntk / torch.max(ntk)
    eigenvalues = torch.linalg.eigvalsh(ntk)
    lmin = eigenvalues[len(eigenvalues) // 2]
    lmax = eigenvalues[-1]
    condition_number = torch.nan_to_num(lmax / lmin, nan=1000000.0)
    return condition_number, lmax, lmin

def lmax(ntk: torch.Tensor):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    return eigenvalues[-1]

def lbulk(ntk: torch.Tensor):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    return eigenvalues[len(eigenvalues) // 2]

def cond(ntk: torch.Tensor):
    eigenvalues = torch.linalg.eigvalsh(ntk)
    lbulk = eigenvalues[len(eigenvalues) // 2]
    lmax = eigenvalues[-1]
    return lmax / lbulk

def pl_condition(grad: torch.Tensor):
    ntk = torch.mm(grad, grad.T)
    singlevalues = torch.linalg.svdvals(ntk)
    singlevalues_sorted, _ = torch.sort(singlevalues, descending=True)
    lmin = singlevalues_sorted[-1]
    lmax = singlevalues_sorted[0]
    condition_number = torch.nan_to_num(lmax / lmin, nan=1000000.0)
    return lmin, lmax, condition_number

def effective_rank(matrix: torch.Tensor):
    singlevalues = torch.linalg.svdvals(matrix)
    singlevalues_norm = singlevalues / torch.sum(singlevalues)
    singlevalues_norm = singlevalues_norm[:sum(singlevalues_norm > 0)]
    rank = torch.sum(-singlevalues_norm * torch.log(singlevalues_norm))
    return rank

def entropy(values: np.array):
    values = values.flatten()
    values_norm = values / np.sum(values)
    e = np.sum(-values_norm * np.log(values_norm))
    return e

def analyse_grad(grad: torch.Tensor):
    avg = 0
    g_0 = grad[0]
    angles = [0]
    for g in grad[1:]:
        cos = g.dot(g_0) / (torch.norm(g, p=2) * torch.norm(g_0, p=2))
        angle = torch.rad2deg(torch.arccos(cos))
        angles.append(angle)
    
    lmin, lmax, condition_number = pl_condition(grad)
    return angles, condition_number

def plot_matrix(matrix, path, colorbar=True, text=True, fontsize=6):
    plt.clf()
    plt.imshow(matrix, cmap='Blues')
    
    if colorbar:
        plt.colorbar()

    if text:
        Y, X = len(matrix), len(matrix[0])
        for y in range(Y):
            for x in range(X):
                plt.text(x, y, '%.1f' % matrix[y][x], ha='center', va='center', fontdict={'size': fontsize})

    plt.savefig(path)

def plot_space(matrix, path):
    eigvals, eigvectors = torch.linalg.eigh(matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Make data
    a, b, c = eigvals
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b',cmap=cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(path)

def plot_to_image(figure):
    plt.savefig('buffer.png', format='png')
    plt.close(figure)
    image = torchvision.io.read_image('buffer.png')
    return image

def render_matrix(matrix, colorbar=True, text=True, log=False):
    figure = plt.figure()
    plt.clf()
    plt.imshow(matrix, cmap='Blues')
    
    if colorbar:
        plt.colorbar()

    if text:
        Y, X = len(matrix), len(matrix[0])
        for y in range(Y):
            for x in range(X):
                plt.text(x, y, '%.1f' % (
                    np.log10(matrix[y][x]) if log else matrix[y][x]
                ), ha='center', va='center', fontdict={'size': 6})

    plt.tight_layout()
    plt.xlabel('Head')
    plt.ylabel('Layer')

    image = plot_to_image(figure)
    
    return image

if __name__ == "__main__":

    S = 0
    A_list = []
    A_cond = []
    for _ in range(1):
        A = torch.randn(10, 100)
        B = torch.randn(10, 100)
        MA = A @ A.T
        MB = B @ B.T

        S += MA
        A_cond.append(cond(MA))
        A_list.append(MA)

        print(cond(MA + MB))

        M = torch.cat([A, B], dim=1)
        MM = M @ M.T
        print(cond(MM))
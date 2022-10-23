from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import math
import os
import torch
import pickle

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

Point = namedtuple('Point', ['x', 'y', 'z'], defaults=[0, 0, 0])

def dist(p1: Point, p2: Point):
    return math.dist(list(p1), list(p2))


def normalize(x, low, high):
    return (x - low) / high


def bound(x, low, high):
    return min(max(x, low), high)


def pdump(x, name, outdir='.'):
    with open(os.path.join(outdir, name), mode='wb') as f:
        pickle.dump(x, f)


def pload(name, outdir='.'):
    with open(os.path.join(outdir, name), mode='rb') as f:
        return pickle.load(f)

def normalize_adjacency_matrix(adj):
    adj += sp.eye(adj.shape[0])  # add self loops
    degree_for_norm = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())  # D^(-0.5)
    adj_hat_csr = degree_for_norm.dot(adj.dot(degree_for_norm))  # D^(-0.5) * A * D^(-0.5)
    adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)

    # to torch sparse matrix
    indices = torch.from_numpy(np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_hat_coo.data)
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(adj_hat_coo.shape), device=device)

    return adjacency_matrix
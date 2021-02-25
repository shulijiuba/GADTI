import numpy as np
import torch as th
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='GRDTI')

    parser.add_argument("--epochs", type=int, default=3000,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=3,
                        help="number of training rounds")
    parser.add_argument("--device", default='cuda',
                        help="cuda or cpu")
    parser.add_argument("--dim-embedding", type=int, default=1000,
                        help="dimension of embeddings")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of iterations in propagation")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--reg_lambda', type=float, default=1,
                        help="reg_lambda")
    parser.add_argument('--patience', type=int, default=6,
                        help='Early stopping patience.')
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Restart Probability")
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge dropout in propagation")

    return parser.parse_args()


def row_normalize(t):
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


def col_normalize(a_matrix, substract_self_loop):
    if substract_self_loop:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    col_sums = a_matrix.sum(axis=0) + 1e-12
    new_matrix = a_matrix / col_sums[np.newaxis, :]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


def l2_norm(t, axit=1):
    t = t.float()
    norm = th.norm(t, 2, axit, True) + 1e-12
    output = th.div(t, norm)
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output



import numpy as np

class GCN:
    def __init__(self, in_features=4, hidden_dim=16):
        self.weights = np.random.randn(in_features, hidden_dim) * 0.01

    def forward(self, node_features, adjacency):
        a_hat = adjacency + np.eye(adjacency.shape[0])
        degree = np.diag(1.0 / np.sqrt(a_hat.sum(axis=1) + 1e-8))
        norm_adj = degree @ a_hat @ degree
        return np.tanh(norm_adj @ node_features @ self.weights)

import numpy as np
import torch
import numba as nb

######kernels used for NMMR and ITR estimation########

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, length_scale=1):
    return torch.exp(-1. * torch.sum((x - y) ** 2, axis=0) / (2 * (length_scale ** 2)))


def calculate_kernel_matrix(dataset, kernel=rbf_kernel, **kwargs):
    tensor = dataset.permute(1, 0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)

def calculate_kernel_matrix_batched(dataset, batch_indices:tuple, kernel=rbf_kernel, **kwargs):
    tensor = dataset.permute(1,0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor1 = tensor1[:, batch_indices[0]:batch_indices[1], :]
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)


def gaussian_kernel(X, X2, sigma):
    res = np.empty((X.shape[0],X2.shape[0]-1),dtype=X.dtype)
    for i in nb.prange(X.shape[0]):
        for j in range(X2.shape[0]-1):
            acc = 0.
            for k in range(X.shape[1]):
                acc += (X[i,k]-X2[k])**2/(2*sigma**2)
            res[i,j] = np.exp(-1*acc)

    return res

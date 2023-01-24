import torch
import numpy as np

######TWO loss functions used for preliminary ITRs estimation########

def h_loss(abs_v, f_re):
    return torch.mean(torch.mul(abs_v, torch.max((1-f_re), torch.tensor(np.zeros((len(f_re),1))))))


def q_loss(abs_v, outcome, f_re):
    return torch.mean(torch.mul(torch.mul(abs_v, outcome), torch.max((1-f_re), torch.tensor(np.zeros((len(f_re),1))))))


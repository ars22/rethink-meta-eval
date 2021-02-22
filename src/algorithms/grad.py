import numpy as np
import torch
import torch.nn.functional as F

quantile_marks = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]

def soft_clip(grad, clip_value=1, slope=0.01):
    assert clip_value > 0 and slope >= 0
    return grad - F.relu((1 - slope) * (grad - clip_value)) + F.relu((1 - slope) * (-clip_value - grad))

def get_grad_norm(grad_list):
    with torch.no_grad():
        grad_norm = 0
        for grad in grad_list:
            grad_norm += grad.norm(2).item() ** 2
        grad_norm = grad_norm ** (1/2)
    return grad_norm

def get_grad_norm_from_parameters(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

def get_grad_entries(grad_list):
    grad_entries = []
    with torch.no_grad():
        for grad in grad_list:
            grad_entries.extend(grad.detach().cpu().flatten().numpy())
    return grad_entries

def get_grad_quantiles(grad_list):
    grad_entries = get_grad_entries(grad_list)
    quantiles = np.quantile(grad_entries, q=quantile_marks)
    return quantiles
import numpy as np
import torch


def randint(low, high=None, size=None):
    """ Credit: https://github.com/pytorch/pytorch/issues/89438#issuecomment-1862363360"""
    if high is None:
        high = low
        low = 0
    if size is None:
        size = low.shape if isinstance(low, torch.Tensor) else high.shape
    return torch.randint(2**63 - 1, size=size, device=high.device) % (high - low) + low


def sample_latents(horizon, num_envs, dimension, steps_min, steps_max, device):
    zbar = torch.zeros((horizon, num_envs, dimension), device=device)
    i = 0
    while i < horizon:
        z = torch.randn((1, num_envs, dimension), device=device)
        reps = torch.randint(steps_min, steps_max+1, size=(1,)).item()
        reps = np.minimum(reps, np.minimum(horizon-i, steps_max+1))
        z = z.repeat_interleave(reps, dim=0)
        zbar[i:reps+i] = z
        i += reps

    return zbar / torch.norm(zbar, dim=-1, keepdim=True)


def sample_easy_latents(n, dimension, device):
    z = torch.normal(torch.zeros([n, dimension], device=device))
    return torch.nn.functional.normalize(z, dim=-1)

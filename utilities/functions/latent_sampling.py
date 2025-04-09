import numpy as np
import torch


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

import torch


@torch.jit.script
def gaussian(x, h, c):
    return torch.exp(-h * (x - c)**2)


@torch.jit.script
def von_mises(x, h, c):
    return torch.exp(h * (torch.cos(x - c) - 1))


@torch.jit.script
def linear(x, h, c):
    return h * (x - c)**2


@torch.jit.script
def inverse_quadratic(x, h, c):
    return 1 / (1 + (h * (x - c)**2))


@torch.jit.script
def multiquadratic(x, h, c):
    return torch.sqrt(1 + (h * (x - c)**2))


@torch.jit.script
def inverse_multiquadratic(x, h, c):
    return 1 / torch.sqrt(1 + (h * (x - c)**2))


@torch.jit.script
def sigmoid(x):
    return torch.sigmoid(x)

import time

import torch
import torch.nn as nn


@torch.jit.script
def gaussian(h, se):
    return torch.exp(-h * se)


@torch.jit.script
def linear(h, se):
    return h * se


@torch.jit.script
def inverse_quadratic(h, se):
    return 1 / (1 + (h * se))


@torch.jit.script
def multiquadratic(h, se):
    return torch.sqrt(1 + (h * se))


@torch.jit.script
def inverse_multiquadratic(h, se):
    return 1 / torch.sqrt(1 + (h * se))


@torch.jit.script
def sigmoid(x):
    return torch.sigmoid(x)


class DMP(nn.Module):

    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        # Simulation parameters
        self.dof = kwargs['actions_num']
        self.num_seqs = kwargs['num_seqs']  # num_envs * num_agents
        self.dt = params['dt'] * params['controlFrequencyInv']
        self.steps_per_seq = params['steps']

        # RBF parameters
        self.kernel = params['kernel']
        self.rbf = eval(self.kernel)
        self.num_rbfs = params['numRBFs']

        # Hyperparameters
        self.weight_scale = params['weight_scale']
        self.hyperparams_scale = params['hyperparameter_scale']
        self.opt_hyperparams = params['optimize_hyperparameters']
        self.target_crossing = params['target_crossing']

        # Determine number of optional and total parameters
        self.num_opt_params = self.dof * (self.opt_hyperparams + self.target_crossing)
        self.num_input_params = self.dof * (self.num_rbfs + 1) + self.num_opt_params

        self.tau = self.dt * self.steps_per_seq

        # Register all tensors as buffers for .to() etc. to work
        temp_spacing = torch.linspace(0, 1, self.num_rbfs)
        alpha_z = torch.ones((self.num_seqs, self.dof)) * self.hyperparams_scale
        dmp_weights = torch.zeros((self.num_seqs, self.dof, self.num_rbfs))

        self.register_buffer('temp_spacing', temp_spacing)
        self.register_buffer('alpha_z', alpha_z)
        self.register_buffer('dmp_weights', dmp_weights)
        self.register_buffer('g', torch.zeros((self.num_seqs, self.dof)))
        self.register_buffer('y0', torch.zeros((self.num_seqs, self.dof)))

        if self.target_crossing:
            self.register_buffer('dg', torch.zeros((self.num_seqs, self.dof)))

    @property
    def alpha_x(self):
        return self.alpha_z / 3

    @property
    def beta(self):
        return self.alpha_z / 4

    @property
    def c(self):
        return torch.einsum('bd, n -> bdn', -self.alpha_x, self.temp_spacing).exp()

    @property
    def h(self):
        # return self.num_rbfs / self._c
        c = self.c
        h = 1 / (c[..., 1:] - c[..., :-1]) ** 2
        return torch.cat((h, h[..., -1:]), dim=-1)

    def initialize(self, y0, parameters):
        dof, n = self.dof, self.num_opt_params
        weights = parameters[..., :-dof-n] * self.weight_scale
        self.dmp_weights = weights.view(-1, dof, self.num_rbfs)
        self.g = parameters[..., -dof-n:-n]
        if self.target_crossing:
            self.dg = parameters[..., -n:-n+dof]
        if self.opt_hyperparams:
            f = torch.nn.ELU(inplace=True)
            # f = torch.nn.ReLU()
            self.alpha_z = f(parameters[..., -dof:]) * self.hyperparams_scale + 2

        self.y0 = y0

    def evaluate(self, y, dy, step):
        t = step[..., None] * self.dt
        # eps = torch.ones(1, device=self.device) * 1e-10
        # b = torch.maximum(1 - self.alpha_x * self.dt / self.tau, eps)
        # x = torch.pow(b, t/self.dt)
        x = torch.exp(-self.alpha_x/self.tau * t)

        # acceleration progression
        if self.num_rbfs == 0:
            forcing = torch.zeros_like(self.y0, device=self.y0.device)
        else:
            sq_error = torch.pow((x[..., None] - self.c), 2)
            at_x = self.rbf(self.h, sq_error)
            fraction = torch.einsum('bdn, bdn -> bd', self.dmp_weights, at_x)
            # fraction = torch.bmm(self.dmp_weights, at_x.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
            forcing = (fraction / at_x.sum(dim=-1)) * x
        if self.target_crossing:
            g_hat = self.g - self.dg * (1 + (self.tau * torch.log(x))/self.alpha_x)
            inner = self.beta * (g_hat - y) + self.tau * (self.dg - dy)
            dz = (1 - x) * self.alpha_z * inner + forcing
        else:
            # Start- and Goal positions may be equal
            # inner = self.g - y - (self.g - self.y0) * x + forcing
            # dz = self.alpha_z * (self.beta * inner - self.tau * dy)

            # Traditional
            inner = self.beta * (self.g - y) - self.tau * dy
            dz = self.alpha_z * inner + (self.g - self.y0) * forcing
        return y + (dy + dz * self.dt / self.tau**2) * self.dt

    def forward(self, parameters, y0, y, dy, step):
        self.initialize(y0, parameters)
        return self.evaluate(y, dy, step)

    # def reset_idx(self, ids, y, parameters):
    #     dof, n = self.dof, self.num_opt_params
    #     self.dmp_params[ids] = parameters[ids]
    #     weights = parameters[ids, :-dof-n] * self.weight_scale
    #     self.dmp_weights[ids] = weights.view(len(ids), dof, self.num_rbfs)
    #     self.g[ids] = parameters[ids, -dof-n:-n]
    #     if self.opt_alpha_x:
    #         f = torch.nn.Softplus()
    #         self.alpha_x[ids] = f(parameters[ids, -n:-n+dof] + 1) * self.alpha_x_scale
    #         self.c[ids] = torch.einsum('bd, n -> bdn', -self.alpha_x[ids], self.temp_spacing).exp()
    #         self.h[ids] = self.num_rbfs / self.c[ids]
    #     if self.opt_hyperparams:
    #         f = torch.nn.Softplus()
    #         self.alpha_z[ids] = f(parameters[ids, -dof:] + 1) * self.hyperparams_scale
    #         self.beta[ids] = self.alpha_z[ids] / 4
    #
    #     self.y0[ids] = y[ids]


if __name__ == '__main__':
    params = {'dt': 1 / 120,
              'controlFrequencyInv': 2,
              'steps': 10,
              'kernel': 'gaussian',
              'numRBFs': 20,
              'weight_scale': 1,
              'alpha_x_scale': 1,
              'hyperparameter_scale': 1,
              'optimize_hyperparameters': True,
              'target_crossing': False
              }
    kwargs = {'actions_num': 9, 'num_seqs': 4}
    dmp = DMP(params, **kwargs)

    dmp.to('cuda:0')
    print('Done.')

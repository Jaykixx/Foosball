from utilities.models.low_level_controllers.dmps.dmp_base import DMPBase
import torch.nn as nn
import numpy as np
import torch


# TODO: Refractor to fit new implementation


class DiscreteDMP(DMPBase):

    def __init__(self, params, **kwargs):
        DMPBase.__init__(self, params, **kwargs)

    @property
    def num_input_params(self):
        # To determine network output shape
        return self.dof * (self.num_rbfs + 1) + self.num_gain_params

    @property
    def c(self):
        # Reference: https://arxiv.org/pdf/2102.03861 page 5
        return torch.einsum('bd, n -> bdn', -self.alpha_x, self.temp_spacing).exp()

    def _compute_canonical_time(self, step):
        if isinstance(step, int):
            t = step * self.dt
        else:
            t = step[..., None] * self.dt
        # eps = torch.ones(1, device=self.device) * 1e-10
        # b = torch.maximum(1 - self.alpha_x * self.dt / self.tau, eps)
        # x = torch.pow(b, t/self.dt)
        x = torch.exp(-self.alpha_x*t/self.tau)
        return x

    def compute_step(self, y, dy, step, **kwargs):
        x = self._compute_canonical_time(step)
        forcing = self._compute_forcing_fn(x)

        # Start- and Goal positions may be equal
        # inner = self.g - y - (self.g - self.y0) * x + forcing
        # dz = self.alpha_z * (self.beta * inner - self.tau * dy)

        # Traditional
        inner = self.beta * (self.g - y) - self.tau * dy
        tau_dz = self.alpha_z * inner + (self.g - self.y0) * x * forcing
        ddy = tau_dz / self.tau ** 2

        return dy + ddy * self.dt

    def integrate(self, steps):
        device = self.y0.device
        history_y = torch.zeros(self.num_seqs, self.dof, steps + 1, device=device)
        history_dy = torch.zeros(self.num_seqs, self.dof, steps + 1, device=device)
        history_ddy = torch.zeros(self.num_seqs, self.dof, steps + 1, device=device)
        history_y[..., 0] = self.y0
        history_dy[..., 0] = self.dy0
        for i in range(steps):
            # canonical progression
            step = torch.ones(self.num_seqs, device=device) * i
            x = self._compute_canonical_time(step)
            # acceleration progression
            forcing = self._compute_forcing_fn(x)
            inner = self.beta * (self.g - history_y[..., i]) - self.tau * history_dy[..., i]
            tau_dz = self.alpha_z * inner + (self.g - self.y0) * x * forcing
            ddy = tau_dz / self.tau**2

            # forward integration
            history_y[..., i + 1] = history_y[..., i] + history_dy[..., i] * self.dt
            history_dy[..., i + 1] = history_dy[..., i] + history_ddy[..., i] * self.dt
            history_ddy[..., i + 1] = ddy
        return history_y, history_dy, history_ddy

    def fit_to_trajectory(self, q, qd, qdd):
        self.to(q.device)
        self.y0[:] = q[0:1]
        self.dy0[:] = qd[0:1]
        self.g[:] = q[-1:]
        self.w_forcing[:] = 0

        f_desired = self.tau ** 2 * qdd - self.alpha_z * (self.beta * (self.g - q) - self.tau * qd)

        t = torch.linspace(0, self.dt * q.shape[0], q.shape[0]).to(q.device)[:, None]
        x = torch.exp(-self.alpha_x/self.tau * t)
        psi_x = self.rbf(x[..., None], self.h, self.c)
        s = x * (self.g - self.y0)

        # aa = torch.sum(s[..., None] * psi_x * f_desired[..., None], dim=0, keepdim=True)
        # bb = torch.sum(torch.pow(s[..., None],2) * psi_x, dim=0, keepdim=True)
        aa = torch.einsum('ln, lnw, ln->nw', s, psi_x, f_desired)
        bb = torch.einsum('ln, lnw, ln->nw', s, psi_x, s)
        self.w_forcing[:] = aa / (bb + 1e-10)
        self.w_forcing[torch.isnan(self.w_forcing)] = 0
        self.w_forcing /= self.weight_scale


if __name__ == '__main__':
    params = {'dt': 1 / 120,
              'controlFrequencyInv': 2,
              'steps': 60,
              'rbfs': {'kernel': 'gaussian', 'numRBFs': 20},
              'gains': {'fixed': True, 'scale': 1},
              'weight_scale': 1
              }
    kwargs = {'actions_num': 6, 'num_seqs': 1}
    dmp = DiscreteDMP(params, **kwargs)

    dmp.to('cuda:0')
    print('Done.')

    n = 20
    k = 230
    dt = 1 / 60
    ax = 1 / 3
    tau = 1
    t = np.linspace(0, k * dt, n)
    c = np.exp(-ax * t)
    h = np.ones(n) * n ** 1.5 / c / (1 / 3)
    x = [1.0]
    for _ in range(k - 1):
        x.append(x[-1] - ax * dt * x[-1] / tau)
    x = np.asarray(x)
    psi_funs = np.exp(-h * (x[:, None] - c) ** 2)
    aa = np.multiply(x.reshape((1, x.shape[0])), psi_funs.T)

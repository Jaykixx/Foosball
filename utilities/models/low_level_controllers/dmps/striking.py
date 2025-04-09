from utilities.models.low_level_controllers.dmps.discrete import DiscreteDMP
import torch


# TODO: Refractor to fit new implementation


class StrikingDMP(DiscreteDMP):

    def __init__(self, params, **kwargs):
        DiscreteDMP.__init__(self, params, **kwargs)

        self.register_buffer('dg', torch.zeros((self.num_seqs, self.dof)))

    @property
    def num_input_params(self):
        return super().num_input_params + self.dof

    def initialize(self, parameters):
        self.dg[:] = parameters[..., -self.dof:]
        DiscreteDMP.initialize(self, parameters[..., :-self.dof])

    def compute_step(self, y, dy, step, **kwargs):
        x = self._compute_canonical_time(step)
        forcing = self._compute_forcing_fn(x)

        # Striking Motion
        g_hat = self.g - self.dg * (1 + (self.tau * torch.log(x)) / self.alpha_x)
        inner = self.beta * (g_hat - y) + self.tau * (self.dg - dy)
        tau_dz = (1 - x) * self.alpha_z * inner + forcing * x
        ddy = tau_dz / self.tau ** 2

        return dy + ddy * self.dt

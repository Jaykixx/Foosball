from utilities.models.low_level_controllers.dmps.dmp_base import DMPBase
from math import pi
import torch


# TODO: Refractor to fit new implementation


class RythmicDMP(DMPBase):

    def __init__(self, params, **kwargs):
        params['rbfs']['kernel'] = 'von_mises'
        DMPBase.__init__(params, **kwargs)

        # Utilizes 1/tau as frequency with 1 cycle per policy step
        # TODO: Add time scaling if needed

    @property
    def num_input_params(self):
        # To determine network output shape
        return self.dof * self.num_rbfs + self.num_gain_params

    @property
    def c(self):
        return 2*pi*self.temp_spacing

    def _compute_canonical_time(self, step):
        t = 2 * pi * step[..., None] * self.dt
        phi = (t * 1 / self.tau) % (2*pi)
        return phi

    def compute_step(self, y, dy, step, **kwargs):
        phi = self._compute_canonical_time(step)
        forcing = self._compute_forcing_fn(phi)

        # Rhythmic Motion
        inner = self.beta * (self.g - y) - self.tau * dy
        tau_dz = self.alpha_z * inner + forcing
        ddy = tau_dz / self.tau ** 2

        return dy + ddy * self.dt

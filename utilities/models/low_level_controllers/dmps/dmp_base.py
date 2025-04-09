from utilities.models.low_level_controllers import LowLevelControllerBase
from utilities.models.low_level_controllers.dmps.basis_functions import *
from abc import abstractmethod
import torch.nn as nn
import torch


class DMPBase(LowLevelControllerBase, nn.Module):

    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        LowLevelControllerBase.__init__(
            self, control_mode='velocity', target_mode='position'
        )

        # Simulation parameters
        self.dof = kwargs['actions_num']
        self.num_seqs = kwargs['num_seqs']  # num_envs * num_agents
        self.dt = params.get('dt', self.dt)
        self.max_steps = params.get('steps', self.max_steps)

        # RBF parameters
        self.num_rbfs = params['rbfs']['numRBFs']
        self.kernel = params['rbfs']['kernel']
        self.rbf = eval(self.kernel)

        # Gain parameters
        self.gains_scale = params['gains']['scale']
        self.fixed_gains = params['gains']['fixed']
        self.num_gain_params = self.dof if not self.fixed_gains else 0

        self.tau_scale = params['tau']['scale']
        if self.dt is None or self.max_steps is None:
            self.tau = 1.0 * self.tau_scale
        else:
            self.tau = self.max_steps * self.dt * self.tau_scale
        self.weight_scale = params['weight_scale']

        # Register all tensors as buffers for .to() etc. to work
        temp_spacing = torch.linspace(0, 1, self.num_rbfs)
        alpha_z = torch.ones((self.num_seqs, self.dof)) * self.gains_scale
        w_forcing = torch.zeros((self.num_seqs, self.dof, self.num_rbfs))

        self.register_buffer('temp_spacing', temp_spacing)
        self.register_buffer('alpha_z', alpha_z)
        self.register_buffer('w_forcing', w_forcing)
        self.register_buffer('y0', torch.zeros((self.num_seqs, self.dof)))
        self.register_buffer('dy0', torch.zeros((self.num_seqs, self.dof)))
        self.register_buffer('g', torch.zeros((self.num_seqs, self.dof)))

    @property
    @abstractmethod
    def num_input_params(self):
        pass

    @property
    def alpha_x(self):
        return self.alpha_z / 3

    @property
    def beta(self):
        return self.alpha_z / 4

    @property
    @abstractmethod
    def c(self):
        pass

    @property
    def h(self):
        # return self.num_rbfs / self._c
        h = 1 / (self.c[..., 1:] - self.c[..., :-1]) ** 2
        return torch.cat((h, h[..., -1:]), dim=-1)

    def set_parameters(self, parameters):
        if not self.fixed_gains:
            n = self.num_gain_params
            func = torch.nn.ELU(inplace=True)
            weights = parameters[..., :-n] * self.weight_scale
            gains = func(parameters[..., -n:]) * self.gains_scale
            self.alpha_z[:] = gains + 1  # must be > 0 and canonical system unstable if close to 0
        else:
            weights = parameters * self.weight_scale

        self.w_forcing[:] = weights.view(-1, self.dof, self.num_rbfs)

    def _compute_forcing_fn(self, x):
        # acceleration progression
        if self.num_rbfs == 0:
            forcing = torch.zeros_like(self.y0, device=self.y0.device)
        else:
            at_x = self.rbf(x[..., None], self.h, self.c)
            fraction = torch.einsum('bdn, bdn -> bd', self.w_forcing*self.weight_scale, at_x)
            forcing = (fraction / at_x.sum(dim=-1))
        return forcing

    @abstractmethod
    def compute_step(self, y, dy, step, **kwargs):
        pass

    def forward(self, parameters, y0, y, dy, step):
        self.initialize(y0, parameters)
        return self.compute_step(y, dy, step)

    def step_controller(self, count):
        y, dy = self.get_robot_states()
        vel_target = self.compute_step(y, dy, count)
        self.apply_control_target(vel_target)

    def set_target(self, target):
        self.y0[:], self.dy0[:] = self.get_robot_states()
        self.g[:] = target

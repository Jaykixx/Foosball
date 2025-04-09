from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import torch_ext

import torch.nn as nn
import numpy as np
import torch


class ValueNetwork(NetworkBuilder.BaseNetwork):

    def __init__(self, params, **kwargs):
        NetworkBuilder.BaseNetwork.__init__(self)

        input_shape = kwargs.pop('input_shape')
        output_shape = kwargs.pop('output_shape')
        mlp_input_shape = kwargs.pop('cnn_bypass_input_shape')
        self.normalize_input = kwargs.get('normalize_input', False)

        self.load(params)

        self.cnn = nn.Sequential()
        self.mlp = nn.Sequential()

        if self.has_cnn:
            if self.permute_input:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)
            cnn_args = {
                'ctype': self.cnn['type'],
                'input_shape': input_shape,
                'convs': self.cnn['convs'],
                'activation': self.cnn['activation'],
                'norm_func_name': self.normalization,
            }
            self.cnn = self._build_conv(**cnn_args)

        # Add to existing input to bypass cnn with some inputs
        mlp_input_shape += self._calc_input_size(input_shape, self.cnn)

        in_mlp_shape = mlp_input_shape
        if len(self.units) == 0:
            out_size = mlp_input_shape
        else:
            out_size = self.units[-1]

        if self.has_rnn:
            raise NotImplementedError

        mlp_args = {
            'input_size': in_mlp_shape,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer
        }

        self.mlp = self._build_mlp(**mlp_args)

        self.mu = torch.nn.Linear(out_size, output_shape)
        self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
        mu_init = self.init_factory.create(**self.space_config['mu_init'])
        self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
        sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

        if self.fixed_sigma:
            self.sigma = nn.Parameter(
                torch.zeros(
                    output_shape, requires_grad=True, dtype=torch.float32
                ), requires_grad=True)
        else:
            self.sigma = torch.nn.Linear(out_size, output_shape)

        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn['initializer'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        mu_init(self.mu.weight)
        if self.fixed_sigma:
            sigma_init(self.sigma)
        else:
            sigma_init(self.sigma.weight)

        if self.normalize_input:
            if isinstance(input_shape, dict):
                self.running_mean_std = RunningMeanStdObs(input_shape)
            else:
                self.running_mean_std = RunningMeanStd(input_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        states = obs_dict.get('rnn_states', None)
        cnn_bypass_obs = obs_dict.get('cnn_bypass_obs', None)

        obs = self.norm_obs(obs)

        if self.has_cnn:
            # for obs shape 4
            # input expected shape (B, W, H, C)
            # convert to (B, C, W, H)
            if self.permute_input and len(obs.shape) == 4:
                obs = obs.permute((0, 3, 1, 2))

        out = obs
        out = self.cnn(out)
        out = out.contiguous().view(out.size(0), -1)

        if cnn_bypass_obs is not None:
            out = torch.cat((out, cnn_bypass_obs), dim=-1)
        if self.has_rnn:
            raise NotImplementedError
        else:
            out = self.mlp(out)

        mu = self.mu_act(self.mu(out))
        if self.fixed_sigma:
            sigma = mu * 0.0 + self.sigma_act(self.sigma)
        else:
            sigma = self.sigma_act(self.sigma(out))

        if self.training:
            sigma = torch.exp(sigma)
            distr = torch.distributions.Normal(mu, sigma)
            result = {
                'rnn_states': states,
                'mus': mu,
                'sigmas': sigma,
                'actions': distr.sample()
            }
        else:
            result = {
                'rnn_states': states,
                'mus': mu,
                'sigmas': sigma,
            }
        return result

    def is_rnn(self):
        return self.has_rnn

    def get_default_rnn_state(self):
        if not self.has_rnn:
            return None
        raise NotImplementedError

    def load(self, params):
        self.units = params['mlp']['units']
        self.initializer = params['mlp']['initializer']
        self.is_d2rl = params['mlp'].get('d2rl', False)
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
        self.activation = params.get('value_activation', 'None')
        self.normalization = params.get('normalization', None)
        self.has_rnn = 'rnn' in params
        self.has_space = 'space' in params
        self.joint_obs_actions_config = params.get('joint_obs_actions', None)

        if self.has_space:
            self.is_multi_discrete = 'multi_discrete'in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous'in params['space']
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False

        if self.has_rnn:
            self.rnn_units = params['rnn']['units']
            self.rnn_layers = params['rnn']['layers']
            self.rnn_name = params['rnn']['name']
            self.rnn_ln = params['rnn'].get('layer_norm', False)
            self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
            self.rnn_concat_input = params['rnn'].get('concat_input', False)

        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']
            self.permute_input = self.cnn.get('permute_input', True)
        else:
            self.has_cnn = False

    def set_state(self, state):
        self.load_state_dict(state['model'])
        if self.normalize_input and 'running_mean_std' in state:
            self.running_mean_std.load_state_dict(state['running_mean_std'])

    def get_state(self):
        state = {
            'model': self.state_dict()
        }
        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        return state

"""
This file implements a wrapper for rl_games\algos_torch\models.py to accomodate
Neural Dynamic Policies.
"""

from rl_games.algos_torch.models import BaseModel, BaseModelNetwork
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import torch_ext

import torch
import numpy as np
import torch.nn as nn

from utilities.models import DMP


class NDP(BaseModel):

    def __init__(self, network):
        super(NDP, self).__init__('a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        @property
        def num_seqs(self):
            return self.a2c_network.num_seqs

        @property
        def dof(self):
            return self.a2c_network.dof

        @property
        def actions_num(self):
            return self.a2c_network.num_input_params

        @property
        def steps_per_seq(self):
            return self.a2c_network.steps_per_seq

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['unnormed_obs'] = input_dict['obs']
            input_dict['unnormed_dmp_init_obs'] = input_dict['dmp_init_obs']
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            input_dict['dmp_init_obs'] = self.norm_obs(input_dict['dmp_init_obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.unnorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                   + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                   + logstd.sum(dim=-1)

    def set_dof(self, dof):
        pass


class NDPA2CBuilder(NetworkBuilder):

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):

        def __init__(self, params, **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)

            self.dmp_layer = DMP(params['dmp'], **kwargs)
            mlp_output_shape = self.dmp_layer.num_input_params
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)
            self.steps_per_seq = self.dmp_layer.steps_per_seq
            self.dof = self.dmp_layer.dof

            self.load(params)

            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

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
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv(**cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

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

            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            # Add multiple heads for critic in accordance with DMP steps
            self.value_heads = torch.nn.Linear(out_size, self.value_size * self.steps_per_seq)
            self.value_act = self.activations_factory.create(self.value_activation)

            self.mu = torch.nn.Linear(out_size, mlp_output_shape)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(self.dof, requires_grad=True, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = torch.nn.Linear(out_size, self.dof)

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

        def forward_dmp(self, obs_dict, parameters):
            y0 = obs_dict['unnormed_dmp_init_obs'][..., :self.dof]
            obs = obs_dict['unnormed_obs']
            seq_step = obs_dict.get('progress_buf', 1)
            seq_step = (seq_step - 1) % self.steps_per_seq
            y, dy = obs[..., :self.dof], obs[..., self.dof:2*self.dof]

            return self.dmp_layer(parameters, y0, y, dy, seq_step)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            dmp_init_obs = obs_dict['dmp_init_obs']
            states = obs_dict.get('rnn_states', None)
            seq_step = obs_dict.get('progress_buf', 1).to(torch.long)
            seq_step = (seq_step - 1) % self.steps_per_seq
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = dmp_init_obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = obs
                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)

                if self.has_rnn:
                    raise NotImplementedError
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

                values = self.value_heads(c_out).T
                idx = seq_step * len(seq_step) + torch.arange(0, len(seq_step), 1, device=seq_step.device, dtype=seq_step.dtype)

                value = self.value_act(values.reshape(-1, 1)[idx])

                mu_params = self.mu_act(self.mu(a_out))
                mu = self.forward_dmp(obs_dict, mu_params)
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
            else:
                a_out = dmp_init_obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.flatten(1)

                c_out = obs
                c_out = self.actor_cnn(c_out)
                c_out = c_out.flatten(1)

                if self.has_rnn:
                    raise NotImplementedError
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.actor_mlp(c_out)

                values = self.value_heads(c_out).T
                idx = seq_step * len(seq_step) + torch.arange(0, len(seq_step), 1, device=seq_step.device, dtype=seq_step.dtype)

                value = self.value_act(values.reshape(-1, 1)[idx])

                if self.central_value:
                    return value, states

                mu_params = self.mu_act(self.mu(a_out))
                mu = self.forward_dmp(obs_dict, mu_params)
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, mu*0 + sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            raise NotImplementedError

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
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

    def build(self, name, **kwargs):
        net = NDPA2CBuilder.Network(self.params, **kwargs)
        return net
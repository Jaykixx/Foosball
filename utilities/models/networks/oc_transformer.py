from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder

from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn as nn
import numpy as np
import torch


class OCTBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self, **kwargs)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            in_size = input_shape[-1]
            self.actor_mlp = self._build_transformer(in_size)
            if self.separate:
                self.critic_mlp = self._build_transformer(in_size)

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            out_size = self.units[-1]
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            mu_init(self.mu.weight)
            if getattr(self.mu, "bias", None) is not None:
                torch.nn.init.zeros_(self.mu.bias)

            if self.is_continuous:
                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(
                        actions_num, requires_grad=True, dtype=torch.float32
                    ), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            value_init = self.init_factory.create(**self.space_config['value_init'])
            value_init(self.value.weight)
            if getattr(self.value, "bias", None) is not None:
                torch.nn.init.zeros_(self.value.bias)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)
            else:
                a_out = self.actor_mlp(obs)
                c_out = a_out

            value = self.value_act(self.value(c_out))
            mu = self.mu_act(self.mu(a_out))

            if self.is_discrete:
                return mu, value, states
            if self.is_multi_discrete:
                raise NotImplementedError  # TODO: Fix
            if self.is_continuous:
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma, value, states


        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)

            self.has_space = 'space' in params
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    raise NotImplementedError
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            # Transformer settings
            self.emb_dim = params['transformer']['emb_dim']
            self.num_heads = params['transformer']['num_heads']
            self.num_blocks = params['transformer']['num_blocks']
            self.dropout_prob = params['transformer']['dropout_prob']
            self.transformer_activation = params['transformer']['activation']
            self.pooling_type = params['transformer']['pooling_type']

            self.central_value = False
            self.has_rnn = False
            self.has_cnn = False

        def _build_transformer(self, input_shape):
            print('build transformer:', input_shape)
            layers = []

            # Initial
            layers.append(nn.Linear(input_shape, self.emb_dim))

            # Transformer
            encoder_layer = TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dim_feedforward=self.emb_dim,
                activation=self.transformer_activation,
                dropout=self.dropout_prob,
                batch_first=True
            )
            layers.extend([
                TransformerEncoder(encoder_layer, self.num_blocks),
                Pooling(self.pooling_type)
            ])

            # Feedforward output layers
            in_size = self.emb_dim
            for unit in self.units:
                layers.append(nn.Linear(in_size, unit))
                layers.append(self.activations_factory.create(self.activation))
                in_size = unit

            return nn.Sequential(*layers)

    def build(self, name, **kwargs):
        net = OCTBuilder.Network(self.params, **kwargs)
        return net


class Pooling(nn.Module):

    def __init__(self, type, **kwargs):
        super(Pooling, self).__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'max':
            return torch.max(x, dim=1)[0]
        elif self.type == 'avg':
            return torch.mean(x, dim=1)

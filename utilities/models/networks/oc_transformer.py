from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder

from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn as nn
import numpy as np
import torch


class OCTBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            # TODO: Construct Transformer Network
            # TODO: Meaning of all params?
            encoder_layer = TransformerEncoderLayer(emb_dim, num_heads,
                                                    emb_dim, device=device,
                                                    dropout=0.1, batch_first=True)
            self.actor_mlp = nn.Sequential(
                nn.Linear(dims[1], emb_dim, device=device),
                TransformerEncoder(encoder_layer, num_blocks),
                nn.Flatten(),
                nn.Linear(dims[0] * emb_dim, emb_dim, device=device),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim, device=device),
                nn.ReLU(),
            )

            if self.separate:
                self.critic_mlp = 0

            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            mu_init(self.mu.weight)

            if self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                          requires_grad=True)
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

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            # TODO: Implement
            if self.separate:
                pass
            else:
                pass

        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params  # TODO: ???
            self.central_value = params.get('central_value', False)  # TODO: ???
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:  # TODO: ???
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
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

            self.has_rnn = False
            self.has_cnn = False

    def _build_transformer(
            self,
            input_shape,
            units,
            activation,
            dense_func,
            norm_only_first_layer=False,
            norm_func_name=None
    ):
        print('build transformer:', input_shape)
        in_size = input_shape
        layers = []
        need_norm = True
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(self.activations_factory.create(activation))

            if not need_norm:
                continue
            if norm_only_first_layer and norm_func_name is not None:
                need_norm = False
            if norm_func_name == 'layer_norm':
                layers.append(torch.nn.LayerNorm(unit))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm1d(unit))
            in_size = unit

        return nn.Sequential(*layers)

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net

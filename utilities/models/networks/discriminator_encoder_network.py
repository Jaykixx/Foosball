from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import torch_ext

import torch.nn as nn
import numpy as np
import torch


class DiscEncNetwork(NetworkBuilder.BaseNetwork):

    def __init__(self, params, **kwargs):
        NetworkBuilder.BaseNetwork.__init__(self)

        input_shape = kwargs.pop('input_shape')

        self.load(params)
        self.discriminator = nn.Sequential()
        self.encoder = nn.Sequential()

        if len(self.units) == 0:
            out_size = input_shape
        else:
            out_size = self.units[-1]

        disc_args = {
            'input_size': input_shape,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': nn.Linear,
            'norm_only_first_layer': self.norm_only_first_layer
        }
        self.disc_mlp = self._build_mlp(**disc_args)

        if self.separate:
            self.enc_mlp = self._build_mlp(**disc_args)

        self.disc_head = nn.Linear(out_size, self.disc_outshape)
        self.disc_activation = self.activations_factory.create(self.disc_activation)
        torch.nn.init.uniform_(self.disc_head.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self.disc_head.bias)

        self.enc_head = nn.Linear(out_size, self.enc_outshape)
        self.enc_activation = self.activations_factory.create(self.enc_activation)
        torch.nn.init.uniform_(self.enc_head.weight, -0.1, 0.1)
        torch.nn.init.zeros_(self.enc_head.bias)

        mlp_init = self.init_factory.create(**self.initializer)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.normalize_input:
            if isinstance(input_shape, dict):
                self.running_mean_std = RunningMeanStdObs(input_shape)
            else:
                self.running_mean_std = RunningMeanStd(input_shape)

    @property
    def device(self):
        return self.disc_head.weight.device

    def _build_sequential_mlp(self,
                              input_size,
                              units,
                              activation,
                              dense_func,
                              norm_only_first_layer=False,
                              norm_func_name=None):
        print('build mlp:', input_size)
        in_size = input_size
        layers = []
        need_norm = True
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(nn.Dropout(self.dropout_prob))
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

    def get_normalized_input(self, obs):
        if self.normalize_input:
            proc_obs = self.running_mean_std(obs)
        else:
            proc_obs = obs
        return proc_obs

    def get_logits(self, obs):
        if self.separate:
            d_out = self.disc_mlp(obs)
            e_out = self.enc_mlp(obs)
        else:
            d_out = self.disc_mlp(obs)
            e_out = self.disc_mlp(obs)

        d_out = self.disc_head(d_out)
        e_out = self.enc_head(e_out)

        return d_out, e_out

    def get_disc_logits(self, obs):
        d_out = self.disc_mlp(obs)
        d_out = self.disc_head(d_out)
        return d_out

    def get_enc_logits(self, obs):
        if self.separate:
            e_out = self.enc_mlp(obs)
        else:
            e_out = self.disc_mlp(obs)

        e_out = self.enc_head(e_out)

        return e_out

    def forward(self, obs):
        proc_obs = self.get_normalized_input(obs)
        d_out, e_out = self.get_logits(proc_obs)

        d_out = self.disc_activation(d_out)
        e_out = self.enc_activation(e_out)

        return d_out, e_out / e_out.norm(dim=-1, keepdim=True)
        # return d_out, torch.nn.functional.normalize(e_out, dim=-1)

    def get_disc_logit_weights(self):
        return torch.flatten(self.disc_head.weight)

    def load(self, params):
        self.separate = params.get('separate', False)
        self.normalization = params.get('normalization', None)
        self.normalize_input = params.get('normalize_input', False)

        # Params
        self.dropout_prob = params.get('dropout_prob', 0.5)

        # MLP
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)

        # Discriminator
        self.disc_activation = params['discriminator'].get('activation', 'sigmoid')
        self.disc_outshape = params['discriminator']['output_shape']

        # Encoder
        self.enc_activation = params['encoder'].get('activation', None)
        self.enc_outshape = params['encoder']['output_shape']

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

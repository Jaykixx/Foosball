from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from rl_games.common import env_configurations
from rl_games.common import vecenv

from utilities.environment.env_configurations import get_extended_env_info
from utilities.models.networks import DiscEncNetwork
from utilities.models.players import A2CPlayer
from utilities.functions.sampling import *

from gym import spaces
import numpy as np
import torch


def ase_player(params):
    level = params['train_level']

    if level == 'high':
        return HighLevelPlayer(params)
    else:
        return LowLevelPlayer(params)


class ASEPlayer(A2CPlayer):

    def __init__(self, params):
        self.latent_dimension = params['config']["latent_dimension"]

        A2CPlayer.__init__(self, params)

        self.horizon = self.env.task._max_episode_length
        self.joint_obs_dimension = self.env_info['joint_observation_space'].shape[0]

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)


class HighLevelPlayer(ASEPlayer):

    def __init__(self, params):
        ASEPlayer.__init__(self, params)
        self.low_level_player = self.load_low_level_networks(params)

        self.apply_dmps = self.low_level_player.apply_dmps
        if self.apply_dmps:
            self.build_dmp_models(params['low_level_params']['dmp'])

    def _get_policy_out_num(self):
        return self.latent_dimension

    def load_low_level_networks(self, params):
        low_params = params['low_level_params']
        env_info = self.env_info.copy()
        env_info['observation_space'] = env_info['joint_observation_space']
        low_params['config']['env_info'] = env_info
        low_params['config']['vec_env'] = self.env
        player = LowLevelPlayer(low_params)

        checkpoint = params['high_level_params']['low_level_checkpoint']
        if checkpoint is not None and checkpoint != '':
            checkpoint = retrieve_checkpoint_path(checkpoint)
            player.restore(checkpoint)

        if player.is_rnn:
            player.init_rnn()

        return player

    def get_action(self, obs, is_determenistic=False):
        actions = ASEPlayer.get_action(self, obs.clone(), is_determenistic)
        normalized_latents = actions / torch.norm(actions, dim=1, keepdim=True)

        low_obs = obs[:, :self.joint_obs_dimension]
        self.low_level_player.current_latents = normalized_latents

        return self.low_level_player.get_action(low_obs, is_determenistic=True)

    def get_batch_size(self, obses, batch_size):
        batch_size = ASEPlayer.get_batch_size(self, obses, batch_size)
        self.low_level_player.has_batch_dimension = self.has_batch_dimension
        self.low_level_player.batch_size = self.batch_size

        return batch_size



class LowLevelPlayer(ASEPlayer):

    def __init__(self, params):
        ASEPlayer.__init__(self, params)

        self.current_latents = None
        self.latent_steps_min = params['config']["latent_steps_min"]
        self.latent_steps_max = params['config']["latent_steps_max"]

    def _get_policy_input_shape(self):
        return (self.obs_shape[0] + self.latent_dimension,)

    def get_action(self, obs, is_determenistic=False):  # to prevent any changes from leaking
        if self.current_latents.ndim == 3:
            # TODO: Fix slicing to avoid for loop
            # latents = self.current_latents[self._env_progress_buffer]
            latents = []
            for i, b in enumerate(self._env_progress_buffer % self.horizon):
                latents.append(self.current_latents[b, i])
            latents = torch.stack(latents, dim=0)
        else:
            latents = self.current_latents
        full_obs = torch.cat((obs, latents), dim=-1)

        return A2CPlayer.get_action(self, full_obs, is_determenistic)

    def model_resets_on_dones(self, all_done_indices):
        self.current_latents = sample_latents(
            horizon=self.horizon,
            num_envs=self.env.num_envs,
            dimension=self.latent_dimension,
            steps_min=self.latent_steps_min,
            steps_max=self.latent_steps_max,
            device=self.device
        )
        A2CPlayer.model_resets_on_dones(self, all_done_indices)

    def env_reset(self, env):
        obs = env.reset()
        obs = self.obs_to_torch(obs)

        indices = torch.arange(
            self.env.num_envs, dtype=torch.int64, device=self.device
        )
        self.model_resets_on_dones(indices)
        return obs

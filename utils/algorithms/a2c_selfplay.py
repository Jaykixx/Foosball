import copy
import random
import torch

from collections import deque

from a2c_base import ContinuousA2CBase


class SelfPlayA2C(ContinuousA2CBase):
    def __init__(self, same_policy_steps, policy_buffer_size,
                 policy_safe_steps, **kwargs):
        super().__init__(**kwargs)
        self.same_policy_steps = same_policy_steps
        self.policy_buffer = deque(maxlen=policy_buffer_size)
        self.policy_safe_steps = policy_safe_steps

        self.current_opponent = None

        self._steps = 0

        self.obs = None  # TODO: Find a better way

    def _invert_obs(self, processed_obs):
        return processed_obs  # TODO: Implement

    def get_opponent_actions(self, obs):
        inverted_obs = self._invert_obs(obs)
        self.current_opponent.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': inverted_obs,
            # 'rnn_states': self.rnn_states # TODO: RNN states
        }
        if self.is_ndp:
            proc_dmp_init_obs = self._preproc_obs(inverted_obs['dmp_init_obs'])
            input_dict['dmp_init_obs'] = proc_dmp_init_obs
            input_dict['progress_buf'] = self._env_progress_buffer

        with torch.no_grad():
            res_dict = self.current_opponent(input_dict)

        return res_dict['actions']

    def env_step(self, actions):
        if self._steps % self.same_policy_steps == 0:
            self.sample_opponent()
        if self._steps % self.policy_safe_steps == 0:
            self.policy_buffer.append(copy.deepcopy(self.model))

        opponent_action = self.get_opponent_actions(self.obs)

        combined_actions = torch.cat((actions, opponent_action), 1)

        self.obs, rewards, dones, infos = super().env_step(combined_actions)
        return self.obs, rewards, dones, infos

    def env_reset(self):
        self.obs = super().env_reset()
        return self.obs

    def sample_opponent(self):
        self.current_opponent = random.choice(self.policy_buffer)

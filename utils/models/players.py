"""
An extension to rl_games\algos_torch\players.py for new policies.
"""

from rl_games.algos_torch.players import PpoPlayerContinuous, rescale_actions
from rl_games.common.tr_helpers import unsqueeze_obs

from utils.models import model_builder

import torch
import time


class A2CPlayer(PpoPlayerContinuous):

    def __init__(self, params):
        self.model_name = params['model']['name']
        self.is_ndp = True if self.model_name == 'ndp' else False
        PpoPlayerContinuous.__init__(self, params)
        if hasattr(self.model, "init_tensors"):
            self.model.init_tensors(self.device)

    @property
    def _env_progress_buffer(self):
        return self.env.progress_buffer

    def load_networks(self, params):
        builder = model_builder.CustomModelBuilder()
        self.config['network'] = builder.load(params)

    def get_action(self, obs, is_determenistic=False):
        processed_obs = self._preproc_obs(obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.states
        }
        if self.is_ndp:
            proc_dmp_init_obs = self._preproc_obs(obs['dmp_init_obs'])
            input_dict['dmp_init_obs'] = proc_dmp_init_obs
            input_dict['progress_buf'] = self._env_progress_buffer

        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0)
            )
        else:
            return current_action

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)
            obs = {
                'obs': obses,
                'dmp_init_obs': obses.clone()
            }

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obs, is_determenistic)

                obses, r, done, info = self.env_step(self.env, action)
                obs['obs'] = obses
                if self.is_ndp:
                    reset_init_obs = (self._env_progress_buffer - 1) % self.model.steps_per_seq == 0
                    obs['dmp_init_obs'][reset_init_obs] = obs['obs'][reset_init_obs]

                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    # if batch_size // self.num_agents == 1 or games_played >= n_games:
                    #     break
                    if games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)
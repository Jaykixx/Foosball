"""
An extension to rl_games\algos_torch\players.py for new policies.
"""

from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions

from utilities.environment.env_configurations import get_extended_env_info
from utilities.models.low_level_controllers.dmps.dmp_builder import DMPBuilder
from utilities.models import model_builder

import torch
import time


class A2CPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        self.env_info = get_extended_env_info(self.env)
        self.joint_obs_dimension = self.env_info['joint_observation_space'].shape[0]
        joint_space = self.env_info['joint_space']
        self.joint_space_low = torch.from_numpy(joint_space.low.copy()).float().to(self.device)
        self.joint_space_high = torch.from_numpy(joint_space.high.copy()).float().to(self.device)
        self.task_has_gripper = self.env_info.get('has_gripper', False)

        self.num_actors = self.config['num_actors']

        self.apply_dmps = False
        if 'dmp' in params:
            self.dmp_config = params['dmp']
            self.apply_dmps = self.dmp_config.get('active', False)
            if self.apply_dmps:
                self.build_dmp_models(params['dmp'])

        self.build_model()
        # if params.get("opponent", False):  # TODO: Selfplay?
        #     self.model.eval()

    @property
    def _env_progress_buffer(self):
        return self.env.progress_buffer

    def _get_policy_input_shape(self):
        return self.obs_shape

    def _get_policy_out_num(self):
        if self.apply_dmps:
            return self.dmp.num_input_params + self.task_has_gripper
        else:
            return self.actions_num

    def build_dmp_models(self, params):
        dmp_builder = DMPBuilder()
        kwargs = {
            'actions_num': self.actions_num - self.task_has_gripper,
            'num_seqs': self.num_actors * self.num_agents,
        }
        self.dmp = dmp_builder.build(params, **kwargs)
        self.dmp.to(self.device)

    def build_model(self):
        policy_input_shape = self._get_policy_input_shape()
        policy_out_num = self._get_policy_out_num()

        build_config = {
            'actions_num': policy_out_num,
            'input_shape': policy_input_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input
        }

        self.model = self.network.build(build_config)
        self.model.to(self.device)
        self.is_rnn = self.model.is_rnn()
        self.model.eval()

    def load_networks(self, params):
        builder = model_builder.CustomModelBuilder()
        self.config['network'] = builder.load(params)

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        return current_action

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def env_step(self, env, actions):
        if self.apply_dmps:
            dmp_parameters = actions[..., :-self.actions_num]
            self.dmp.set_parameters(dmp_parameters)
            self.env.set_low_level_controller(self.dmp)  # TODO: Refractor for multi-dmp-variant
            actions = actions[..., -self.actions_num:]

        actions = self.preprocess_actions(actions)
        obs, task_rewards, dones, infos = BasePlayer.env_step(self, env, actions)

        return obs, task_rewards, dones, infos

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def model_resets_on_dones(self, all_done_indices):
        if self.is_rnn:
            for s in self.states:
                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

    def env_reset(self, env):
        obses = BasePlayer.env_reset(self, env)
        self.obs = obses.clone()  # Required for DMP support in env_step
        return obses

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
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
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)

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
                    self.model_resets_on_dones(all_done_indices)

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

    def evaluate(self):
        n_games = 8192
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
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
        all_rewards = []
        all_steps = []
        all_successes = []
        all_failures = []
        all_timeouts = []
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            # Check each env only once otherwise we have bias towards failures
            already_counted = torch.zeros(self.env.task._num_envs, dtype=torch.bool)
            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)

                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)

                if done_count > 0:
                    done_indices = done_indices[~already_counted[done_indices]]
                    games_played += len(done_indices)
                    already_counted[done_indices] = True
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    all_rewards.append(cr[done_indices])
                    all_steps.append(steps[done_indices])

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    all_successes.append(info['Successes'][done_indices])
                    all_failures.append(info['Failures'][done_indices])
                    all_timeouts.append(info['Timeouts'][done_indices])

                    if (already_counted == False).sum() == 0:
                        break
                    if games_played >= n_games:
                        break

        successes = torch.concatenate(all_successes)
        failures = torch.concatenate(all_failures)
        timeouts = torch.concatenate(all_timeouts)
        if self.print_stats:
            total_games = games_played * n_game_life
            print(
                '\nav reward: ', sum_rewards / total_games,
                '\nav steps: ', sum_steps / total_games,
                '\nav success rate: ', successes.sum().item() / total_games,
                '\nav failure rate: ', failures.sum().item() / total_games,
                '\nav timeout rate: ', timeouts.sum().item() / total_games
            )

        results = {
            'successes': successes,
            'failures': failures,
            'timeouts': timeouts,
            'rewards': torch.concatenate(all_rewards, dim=0),
            'steps': torch.concatenate(all_steps, dim=0),
            'games_played': games_played * n_game_life
        }
        return results

    def run_physical_env(self):
        is_deterministic = self.is_deterministic

        obs = self.env.reset()

        done = False
        while not done:
            clk = time.perf_counter()
            action = self.get_action(obs, is_deterministic)  # [0]
            # action = 0.3*(torch.rand(7, device=self.device) - 0.5)
            inference_time = time.perf_counter() - clk
            print("Inference Time: {:.2f} ms".format(inference_time * 1000))

            obs, done = self.env.step(action)

        return

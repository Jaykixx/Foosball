import os
import copy
import torch

from environments.foosball.foosball_base import FoosballTask

from utilities.custom_runner import CustomRunner as Runner
import time


class FoosballSelfPlay(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_actions"):
            # Defines action space for AI
            self._num_actions = 8
        if not hasattr(self, "_dof"):
            # Defines action space for task - Only different for selfplay
            self._dof = 2 * self._num_actions
        if not hasattr(self, "_num_task_observations"):
            # Ball + Opponents (pos + vel)
            self._num_task_observations = 2 * self.num_actions + 4

            # Second option only relevant for playtest on real system due to limited obs
            # Only applicable to standard obs since object centric will require same features per object
            # Ball + Opponents Prismatic Joints (pos + vel)
            # self._num_task_observations = self.num_actions + 4

        if self._num_task_observations == (2 * self.num_actions + 4):
            self._full_opponent_obs = True
        else:
            self._full_opponent_obs = False

        FoosballTask.__init__(self, name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

        self.num_opponents = self._task_cfg["env"].get("num_opponents", 1)
        self.opponents_obs_ranges = [
            i * self._num_envs // self.num_opponents for
            i in range(self.num_opponents + 1)
        ]

        # on reset there are no observations available
        self._full_actions = self._duplicate_actions

        # introduce buffers to catch stagnating game states
        n_history = 60
        self.ball_vel_history = torch.zeros((self.num_envs, n_history), device=self.device)

        self.opponents = None

    def add_opponent_action(self, actions):
        op_actions = tuple([
            torch.atleast_2d(
                self.opponents[i].get_action(
                    self.inv_obs_buf[
                        self.opponents_obs_ranges[i]:self.opponents_obs_ranges[i + 1],
                        ...
                    ], is_deterministic=True
                ).detach()
            )
            for i in range(self.num_opponents)
        ])
        return torch.cat((actions, torch.cat(op_actions, 0)), 1)

    def cleanup(self) -> None:
        super().cleanup()
        self.inv_obs_buf = torch.zeros_like(self.obs_buf)

    def get_standard_observations(self) -> dict:
        dof_pos = self._robots.get_joint_positions(joint_indices=self.active_joint_dofs, clone=False)
        dof_vel = self._robots.get_joint_velocities(joint_indices=self.active_joint_dofs, clone=False)
        dof_pos_w = dof_pos[:, :self.num_actions]
        dof_pos_b = dof_pos[:, self.num_actions:]
        dof_vel_w = dof_vel[:, :self.num_actions]
        dof_vel_b = dof_vel[:, self.num_actions:]

        # Observe game ball in x-, y-axis
        ball_w_pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = ball_w_pos[:, :2] - self._env_pos[:, :2]
        ball_vel = self._balls.get_velocities(clone=False)[:, :2]

        if self._full_opponent_obs:
            self.obs_buf = torch.cat(
                (dof_pos_w, dof_vel_w, dof_pos_b, dof_vel_b, ball_pos, ball_vel), dim=-1
            )

            self.inv_obs_buf = torch.cat(
                (dof_pos_b, dof_vel_b, dof_pos_w, dof_vel_w, -ball_pos, -ball_vel), dim=-1
            ).clone()
        else:
            # num_actions is always even in selfplay (2 Joints per rod)
            half_obs = int(self.num_actions // 2)
            self.obs_buf = torch.cat(
                (dof_pos_w, dof_vel_w, dof_pos_b[:, :half_obs], dof_vel_b[:, :half_obs], ball_pos, ball_vel), dim=-1
            )

            self.inv_obs_buf = torch.cat(
                (dof_pos_b, dof_vel_b, dof_pos_w[:, :half_obs], dof_vel_w[:, :half_obs], -ball_pos, -ball_vel), dim=-1
            ).clone()

    @staticmethod
    def sort_obj_centric_obs(obs, inverse: bool = False):
        x_sorted, sorted_idx = torch.sort(obs[0, :, 3], descending=not inverse)
        unique_x = torch.unique(x_sorted)
        for x in unique_x:
            fig_idx = sorted_idx[x_sorted == x]
            y_sort_idx = torch.sort(obs[0, fig_idx, 4], descending=False).indices
            sorted_idx[x_sorted == x] = fig_idx[y_sort_idx]
        return obs[:, sorted_idx]

    def get_obj_centric_observations(self):
        obj_obs = {
            'player_obs': [],
            'opponent_obs': [],
        }
        inv_obj_obs = {  # Contains inverted obs for opponent query
            'player_obs': [],  # Here used for opponent
            'opponent_obs': [],  # Here used for player
        }
        for name, value in self.active_rods.items():
            # TODO: Rescale to table size
            sign = -1 if 'W' in name else 1  # Joints for black are mirrored so signs are needed

            fig_tpos = self.robot.figure_positions[name][None].repeat_interleave(self.num_envs, 0)
            fig_tpos[:, 1] += sign * self._robots.get_joint_positions(joint_indices=[value['pris_id']], clone=False)

            fig_rpos = sign * self._robots.get_joint_positions(joint_indices=[value['rev_id']], clone=False)
            fig_rpos = fig_rpos[..., None].repeat_interleave(fig_tpos.shape[-1], -1)

            fig_tvel = torch.zeros_like(fig_tpos)
            fig_tvel[:, 1] = sign * self._robots.get_joint_velocities(joint_indices=[value['pris_id']], clone=False)

            fig_rvel = sign * self._robots.get_joint_velocities(joint_indices=[value['rev_id']], clone=False)
            fig_rvel = fig_rvel[..., None].repeat_interleave(fig_tvel.shape[-1], -1)

            one_hot_encoding = torch.zeros((self.num_envs, self._num_obj_types, fig_tpos.shape[-1]), device=self.device)
            inv_one_hot_encoding = torch.zeros_like(one_hot_encoding)
            if 'W' in name:
                rod_idx = self.robot.rod_paths_W.index('White/' + name)
                if hasattr(self, 'white_rods_mask'):
                    mask = self.white_rods_mask[:, rod_idx]
                    one_hot_encoding[mask, 0] = 1
                    inv_one_hot_encoding[mask, 1] = 1  # Register as Black for opponent
                else:
                    one_hot_encoding[:, 0] = 1
                    inv_one_hot_encoding[:, 1] = 1  # Register as Black for opponent
            elif 'B' in name:
                rod_idx = self.robot.rod_paths_B.index('Black/' + name)
                if hasattr(self, 'black_rods_mask'):
                    mask = self.black_rods_mask[:, rod_idx]
                    one_hot_encoding[mask, 1] = 1
                    inv_one_hot_encoding[mask, 0] = 1  # Register as white for opponent
                else:
                    one_hot_encoding[:, 1] = 1
                    inv_one_hot_encoding[:, 0] = 1  # Register as white for opponent

            fig_obs = torch.cat((
                one_hot_encoding, fig_tpos, fig_rpos, fig_tvel, fig_rvel,
            ), dim=1).transpose(1, 2)

            # Order of objects irrelevant for object centric transformers
            #   so instead we keep object order and only switch perspective
            inv_fig_obs = torch.cat((
                inv_one_hot_encoding, -fig_tpos, -fig_rpos, -fig_tvel, -fig_rvel,
            ), dim=1).transpose(1, 2)

            if 'W' in name:
                obj_obs['player_obs'].append(fig_obs)
                inv_obj_obs['opponent_obs'].append(inv_fig_obs)
            elif 'B' in name:
                obj_obs['opponent_obs'].append(fig_obs)
                inv_obj_obs['player_obs'].append(inv_fig_obs)

            # obj_obs.append(fig_obs)
            # inv_obj_obs.append(inv_fig_obs)

        obj_obs['player_obs'] = self.sort_obj_centric_obs(
            torch.cat(obj_obs['player_obs'], dim=1)
        )
        obj_obs['opponent_obs'] = self.sort_obj_centric_obs(
            torch.cat(obj_obs['opponent_obs'], dim=1), inverse=True
        )

        inv_obj_obs['player_obs'] = self.sort_obj_centric_obs(
            torch.cat(inv_obj_obs['player_obs'], dim=1)
        )
        inv_obj_obs['opponent_obs'] = self.sort_obj_centric_obs(
            torch.cat(inv_obj_obs['opponent_obs'], dim=1), inverse=True
        )

        # goals
        obj_obs['player_goal'] = torch.zeros((self.num_envs, 1, self._num_obj_features + self._num_obj_types), device=self.device)
        obj_obs['opponent_goal'] = torch.zeros_like(obj_obs['player_goal'])
        inv_obj_obs['player_goal'] = torch.zeros_like(obj_obs['player_goal'])
        inv_obj_obs['opponent_goal'] = torch.zeros_like(obj_obs['player_goal'])

        obj_obs['player_goal'][..., self._num_obj_types - 3] = 1
        obj_obs['opponent_goal'][..., self._num_obj_types - 2] = 1
        inv_obj_obs['player_goal'][..., self._num_obj_types - 3] = 1
        inv_obj_obs['opponent_goal'][..., self._num_obj_types - 2] = 1

        obj_obs['player_goal'][..., self._num_obj_types] = 0.6
        obj_obs['opponent_goal'][..., self._num_obj_types] = -0.6
        inv_obj_obs['player_goal'][..., self._num_obj_types] = 0.6
        inv_obj_obs['opponent_goal'][..., self._num_obj_types] = -0.6

        # ball
        ball_obs = torch.zeros((self.num_envs, self._num_obj_features + self._num_obj_types), device=self.device)
        ball_pos, ball_vel = self.get_ball_observation()
        ball_obs[..., self._num_obj_types-1] = 1
        # zero ball pose
        # ball_obs[..., self._num_obj_types:self._num_obj_types+2] = ball_pos
        # ball_obs[..., -3:-1] = ball_vel
        inv_ball_obs = ball_obs.clone()
        inv_ball_obs[..., self._num_obj_types:] *= -1
        obj_obs['ball'] = ball_obs[:, None]
        inv_obj_obs['ball'] = inv_ball_obs[:, None]

        obs = torch.cat(list(obj_obs.values()), dim=1)
        inv_obs = torch.cat(list(inv_obj_obs.values()), dim=1)

        # Center obs around ball
        if self.ball_relative_obs:
            obs[:, :-1, self._num_obj_types:self._num_obj_types+2] -= ball_pos[:, None]
            inv_obs[:, :-1, self._num_obj_types:self._num_obj_types+2] += ball_pos[:, None]

            obs[:, :-1, -3:-1] -= ball_vel[:, None]
            inv_obs[:, :-1, -3:-1] += ball_vel[:, None]

        if self.flatten_obs:
            obs = obs.flatten(start_dim=1)
            inv_obs = inv_obs.flatten(start_dim=1)

        # °/s to rad/s
        obs[..., -1] *= torch.pi / 180

        self.obs_buf = obs
        self.inv_obs_buf = inv_obs

    def get_observations(self) -> dict:
        if self.object_centric_obs:
            self.get_obj_centric_observations()
        else:
            self.get_standard_observations()

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf,
            }
        }

        if self.capture:
            self.capture_image()
        return observations

    def _order_joints(self) -> list:
        joints = self.robot.dof_paths_W + self.robot.dof_paths_B
        active_joint_dofs = []
        for j in joints:
            active_joint_dofs.append(self._robots.get_dof_index(j))
        return active_joint_dofs

    def post_reset(self):
        # first half of actions are white, second are black
        self.active_joint_dofs = self._order_joints()
        super().post_reset()

    def reset(self):
        if self.opponents is None:
            self.create_opponent(self._cfg['train'])
        super().reset()

    def reset_idx(self, env_ids):
        FoosballTask.reset_idx(self, env_ids)
        self.ball_vel_history[env_ids] = 0

    def create_opponent(self, config) -> None:
        r = Runner()
        r.load(config)
        # create opponents in eval mode
        r.params["opponent"] = True

        self.opponents = [r.create_player() for _ in range(self.num_opponents)]
        if config['params']['load_checkpoint']:
            for agent in self.opponents:
                agent.restore(config['params']['load_path'])

    def prepare_opponent(self):
        self._full_actions = self.add_opponent_action

    def full_actions(self, actions):
        return self._full_actions(actions)

    @staticmethod
    def _duplicate_actions(actions):
        return torch.cat((actions, actions), 1)

    def update_weights(self, indices, weights):
        for i in indices:
            self.opponents[i%self.num_opponents].set_weights(weights)

    def detect_stagnating_games(self, timeouts):
        ball_vel = self._balls.get_velocities(clone=True)[:, :2]
        ball_vel = torch.norm(ball_vel, dim=-1)
        self.ball_vel_history = torch.cat((self.ball_vel_history[:, 1:], ball_vel[:, None]), dim=-1)

        # Only consider games that have been running for more than considered
        # horizon and are not already in timeouts
        valid_envs = self.progress_buf >= self.ball_vel_history.shape[-1]
        valid_envs = torch.min(~timeouts, valid_envs)
        stagnating = (self.ball_vel_history < 5e-2).all(dim=-1)  # No movement across horizon
        inaction = torch.min(valid_envs, stagnating)

        penalize = torch.min(self.obs_buf[:, :, 0] == 1, torch.abs(self.obs_buf[:, :, self._num_obj_types]) < 0.02).sum(dim=-1, dtype=torch.bool)

        return inaction, ball_vel, penalize & inaction

    def _calculate_metrics(self):
        wins, losses, timeouts = super()._calculate_metrics()

        pos = self._balls.get_world_poses(clone=False)[0]
        ball_pos = pos - self._env_pos

        # Optional Reward: Ball near opponent goal
        # self.rew_buf += self._dist_to_goal_reward(ball_pos)

        # Optional Reward: Regularization of actions
        # self.rew_buf += 0.1 * self._compute_action_regularization()

        # Optional Reward: Pull figures to ball
        # self.rew_buf += self._fig_to_ball_reward(ball_pos)

        # Detect and punish inaction
        inaction, ball_vel, penalize = self.detect_stagnating_games(timeouts)
        self.rew_buf[penalize] = -self.stagnation_penalty
        self.reset_buf = torch.max(self.reset_buf, inaction)

        # Log mean and standard deviation of ball speed to detect stagnating games
        if self.reset_buf.sum() > 0:
            self.extras["Stagnation Rate"] = inaction.sum() / self.reset_buf.sum()
        else:
            self.extras["Stagnation Rate"] = 0.0
        self.extras["Ball Velocity Avg"] = ball_vel.mean()
        self.extras["Ball Velocity Std"] = ball_vel.std()

        return wins, losses, timeouts
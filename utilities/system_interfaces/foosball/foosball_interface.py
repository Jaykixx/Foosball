from utilities.system_interfaces.foosball.foosball_connector import FoosballConnector
from utilities.system_interfaces.state_estimation.object_detector import ObjectDetector
from utilities.system_interfaces.base_interface import BaseInterface
from utilities.models.kalman_filter import KalmanFilter

from omegaconf import OmegaConf
from gym import spaces
import numpy as np
import cv2 as cv
import torch
import time


class FoosballInterface(BaseInterface):

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg['rl_device']

        self.sys_cfg = self.cfg['system_config']
        self.settings = self.sys_cfg['system']['settings']
        self._table = self.sys_cfg['system']['table_config']
        self.detector = ObjectDetector(self.sys_cfg['object_detection'])

        self._active_drives = self.settings['active_drives']
        self._num_actions = len(self._active_drives)

        self._actuated_rods = self.settings['actuated_rods']
        self._opponent_rods = self.settings['opponent_rods']
        self._passive_rods = self.settings['passive_rods']

        all_rods = self._actuated_rods + self._opponent_rods + self._passive_rods
        self._observable_rods = {key: v for key, v in self._table['figures'].items() if key in all_rods}

        self._observe_joints = self.settings.get('observe_joints', True)
        self._prediction_steps = self.settings.get('prediction_steps', 0)

        # Determine observation space and kalman dimensions
        if self._observe_joints:
            passive_obs = len(self._passive_rods)
            opponent_obs = len(self._opponent_rods)
        else:
            # TODO: Check with config file
            passive_obs = sum([v for key, v in self._table['figures'].items() if key in self._passive_rods])
            opponent_obs = sum([v for key, v in self._table['figures'].items() if key in self._opponent_rods])
            # TODO: Check if detector can detect opponents

        ball_obs = 2  # Only position is detected, velocities in kalman state
        self._num_task_observations = 2*ball_obs + 2*opponent_obs + passive_obs

        BaseInterface.__init__(self)
        self.kalman_ball = KalmanFilter(ball_obs, self._num_envs, self.device)
        self.kalman_opponents = KalmanFilter(opponent_obs, self._num_envs, self.device)

    @property
    def _detect_only_ball(self):
        return self.detector.object_classes == [1]

    def get_system_connector(self):
        return FoosballConnector(self.sys_cfg, device=self.device)

    def shutdown(self):
        BaseInterface.shutdown(self)
        # TODO: Add evaluation of times and images

    def reset(self):
        self.clk = time.perf_counter()
        self.step(torch.zeros(self._num_actions, device=self.device))

    def step(self, actions):
        self.sys_connector.apply_actions(actions)
        done = False  # TODO: Compute
        return done

    def _sort_figure_pos(self, fig_pos):
        if fig_pos is None:
            return  # Skip and kalman filter bridges gap

        if len(fig_pos) < sum(self._observable_rods.values()):
            # TODO: Catch cases with partially missing detections!
            # Find close groups to check where detection is missing?
            return  # Currently skips and kalman filter bridges gap

        # sort figures by x position to assign to rods
        sorted_idx = torch.argsort(fig_pos[:, 0, 0])
        sorted_by_x = fig_pos[sorted_idx]

        # Slice obs according to figure numbers to compute joint positions
        start_idx = 0
        sorted_by_xy = {}
        for k, v in self._observable_rods.items():
            if k not in self._actuated_rods:
                n = len(v)
                if self._observe_joints:
                    sorted_by_xy[k] = torch.mean(sorted_by_x[start_idx:start_idx + n], dim=0)[1:2]
                else:
                    subset = sorted_by_x[start_idx:start_idx + n]
                    sorted_idx = torch.argsort(subset[:, 1, 0])
                    sorted_by_xy[k] = subset[sorted_idx][:, 1]
            start_idx += v
        return sorted_by_xy

    def get_task_observation(self):
        result = self.detector.get_observation()
        positions = result['positions']
        times = result['times']

        clk = time.perf_counter()
        ball_pos = positions['ball']
        self.kalman_ball.predict()
        if ball_pos is not None:
            self.kalman_ball.correct(ball_pos)
        ball_obs = self.kalman_ball.future_predictions(self._prediction_steps)
        # Convert velocities from m/frame to m/s
        dt = time.perf_counter() - self.clk
        self.clk = time.perf_counter()
        ball_obs[self.kalman_ball.num_obs:] /= dt

        task_obs = ball_obs
        if not self._detect_only_ball:
            fig_pos = self._sort_figure_pos(positions['figures'])
            if len(self._passive_rods) > 0:
                pa_obs = torch.concatenate(
                        [fig_pos[k] for k in self._passive_rods], dim=-1
                    )
                task_obs = torch.cat((pa_obs, task_obs), dim=-1)

            self.kalman_opponents.predict()
            if fig_pos is not None and len(self._opponent_rods) > 0:
                op_pos = torch.concatenate(
                        [fig_pos[k] for k in self._opponent_rods], dim=-1
                    )
                self.kalman_opponents.correct(op_pos)
                op_obs = self.kalman_opponents.future_predictions(self._prediction_steps)
                op_obs[self.kalman_opponents.num_obs:] /= dt
                task_obs = torch.cat((op_obs, task_obs), dim=-1)

        times['process_obs_time'] = time.perf_counter() - clk
        return task_obs, times

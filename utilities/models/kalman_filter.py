import torch
import numpy as np


class KalmanFilter:
    def __init__(self, num_obs, num_envs, device='cuda:0'):
        self.num_state, self.num_obs = 2*num_obs, num_obs
        self.num_envs = num_envs
        self.device = device

        self.state = torch.zeros(
            self.num_envs, self.num_state, device=self.device
        )

        # State transition matrix
        stm = torch.eye(self.num_state, device=self.device)
        stm += torch.diag(torch.ones(self.num_obs, device=self.device), self.num_obs)
        self.stm = stm[None].repeat_interleave(self.num_envs, dim=0)

        # Batch Identity matrix
        eye = torch.eye(n=self.num_state, device=self.device)
        self.eye = eye[None].repeat_interleave(self.num_envs, dim=0)

        # Covariance of the system state
        self.eps = self.eye.clone()

        # Process noise
        self.eta = self.eye.clone()

        # Measurement observation matrix
        h = torch.eye(self.num_obs, self.num_state, device=self.device)
        self.h = h[None].repeat_interleave(self.num_envs, dim=0)
        
        # State uncertainty
        var = torch.eye(n=self.num_obs, device=self.device)
        self.var = 0.01 * var[None].repeat_interleave(self.num_envs, dim=0)

        # Kalman matrix
        self.k = torch.zeros(
            self.num_envs, self.num_state, self.num_obs, device=self.device
        )

    def reset_idx(self, env_ids):
        n = len(env_ids)
        
        self.state[env_ids] = torch.zeros(
            n, self.num_state, device=self.device
        )

        eye = torch.eye(n=self.num_state, device=self.device)
        self.eps[env_ids] = eye[None].repeat_interleave(n, dim=0)

        self.k[env_ids] = torch.zeros(
            n, self.num_state, self.num_obs, device=self.device
        )

    def predict(self):
        """
        Make a prediction for the actual state from the previous state.
        :return:
        """

        # compute the state prediction
        self.state = torch.einsum("bni, bi -> bn", self.stm, self.state)

        # compute the prediction of the covariance P
        self.eps = self.stm @ self.eps @ self.stm.transpose(-2, -1) + self.eta

    def correct(self, z):
        """
        Correct the predicted state with a measurement z.
        :param z: measured state
        :return:
        """
        inv = torch.linalg.inv(self.h @ self.eps @ self.h.transpose(-2, -1) + self.var)
        self.k = self.eps @ self.h.transpose(-2, -1) @ inv
        diff = z - torch.einsum("bni, bi -> bn", self.h, self.state)
        self.state = self.state + torch.einsum("bin, bn -> bi", self.k, diff)
        self.eps = (self.eye - self.k @ self.h) @ self.eps

    def future_predictions(self, steps):
        # TODO: add collisions with the wall (Only affects simulation)
        # check validity of position find out with which wall the ball collides
        # torch.matrix_power(self.stm, steps) @ self.state ? valid?
        future_state = self.state.clone()
        for i in range(steps):
            future_state = torch.einsum("bni, bi -> bn", self.stm, future_state)
        return future_state

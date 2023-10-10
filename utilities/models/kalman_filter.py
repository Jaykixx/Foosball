import torch
import numpy as np


class KalmanFilter:
    def __init__(self, n_s, n_o, num_envs, device='cuda:0'):
        self.n_s, self.n_o = n_s, n_o
        self.num_envs = num_envs
        self.device = device
        self.initialize()

    def initialize(self):
        # contains the state of the system, with n-dimensions
        self.state = torch.zeros(self.num_envs, self.n_s, 1, device=self.device)

        # contains a possible future state
        self.future_state = torch.zeros(self.num_envs, self.n_s, 1, device=self.device)

        # contains the state transition matrix in our case it is often the identity matrix
        self.F = torch.eye(self.n_s, device=self.device) + torch.diag(torch.ones(self.n_o, device=self.device), self.n_o)
        self.F = self.F[None].repeat_interleave(self.num_envs, dim=0)

        # contains a state transition to a future state
        self.F_future = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(self.num_envs, dim=0)

        # contains the covariance of the systems state, by identity
        self.P = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(self.num_envs, dim=0)

        # contains the process noise, initalized by identity
        self.Q = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(self.num_envs, dim=0)

        # contains the measurement observation matrix
        H = np.eye(self.n_o, self.n_s).astype(np.float32)
        self.H = torch.from_numpy(H).to(self.device)[None].repeat_interleave(self.num_envs, dim=0)
        # self.H = torch.zeros(self.n_o, self.n_s).to(self.device)

        # contains the state uncertainty
        self.R = torch.eye(n=self.n_o, device=self.device)[None].repeat_interleave(self.num_envs, dim=0) * 0.01

        # contains the Kalman matrix
        self.K = torch.zeros(self.num_envs, self.n_s, self.n_o, device=self.device)

        # contains the Identity Matrix
        self.I = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(self.num_envs, dim=0)

    def reset_idx(self, env_ids):
        n = len(env_ids)

        self.state[env_ids] = torch.zeros(n, self.n_s, 1, device=self.device)
        self.future_state[env_ids] = torch.zeros(n, self.n_s, 1, device=self.device)
        F = torch.eye(self.n_s, device=self.device) + torch.diag(torch.ones(self.n_o, device=self.device), self.n_o)
        self.F[env_ids] = F[None].repeat_interleave(n, dim=0)
        self.F_future[env_ids] = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(n, dim=0)
        self.P[env_ids] = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(n, dim=0)
        self.Q[env_ids] = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(n, dim=0)
        H = np.eye(self.n_o, self.n_s).astype(np.float32)
        self.H[env_ids] = torch.from_numpy(H).to(self.device)[None].repeat_interleave(n, dim=0)
        self.R[env_ids] = torch.eye(n=self.n_o, device=self.device)[None].repeat_interleave(n, dim=0) * 0.01
        self.K[env_ids] = torch.zeros(n, self.n_s, self.n_o, device=self.device)
        self.I[env_ids] = torch.eye(n=self.n_s, device=self.device)[None].repeat_interleave(n, dim=0)


    def predict(self):
        """
        Make a prediction for the actual state from the previous state.
        :return:
        """

        # compute the state prediction
        self.state = self.F @ self.state

        # compute the prediction of the covariance P
        self.P = self.F @ self.P @ self.F.transpose(-2, -1) + self.Q

    def correct(self, z):
        """
        Correct the predicted state with a measurement z.
        :param z: measured state
        :return:
        """
        inv = torch.linalg.inv(self.H @ self.P @ self.H.transpose(-2, -1) + self.R)
        self.K = self.P @ self.H.transpose(-2, -1) @ inv
        self.state = self.state + self.K @ (z - self.H @ self.state)
        self.P = (self.I - self.K @ self.H) @ self.P

    def future_predictions(self, steps):
        self.future_state = torch.matrix_power(self.F, steps) @ self.state
        # TODO: add collisions with the wall
        # checK validity of position find out with which wall the ball collides
        # update state
        pass

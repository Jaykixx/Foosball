from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict
import numpy as np
import torch
import gym


class ReplayBuffer:

    def __init__(self, tensor_dict, capacity, device):
        self.device = device
        self.capacity = capacity
        self.base_shape = (capacity,)
        self.idx = 0
        self.full = False

        self.tensor_dict = {}
        for k, v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(v), dtype=np.float32), self.base_shape)

    def _create_tensor_from_space(self, space, base_shape):
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)
        if type(space) is gym.spaces.Discrete:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype= dtype, device=self.device)
        if type(space) is gym.spaces.Tuple:
            '''
            assuming that tuple is only Discrete tuple
            '''
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(base_shape +(tuple_len,), dtype=dtype, device=self.device)
        if type(space) is gym.spaces.Dict:
            t_dict = {}
            for k, v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict

    def add(self, tensor_dict):
        num_adds = next(iter(tensor_dict.values())).shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_adds)
        overflow = num_adds - remaining_capacity
        self.full = remaining_capacity < num_adds
        for k, v in tensor_dict.items():
            if self.full:
                self.tensor_dict[k][0:overflow] = v[-overflow:]
            self.tensor_dict[k][self.idx:self.idx+remaining_capacity] = v[:remaining_capacity]

        self.idx = (self.idx + num_adds) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = torch.randint(0,
                             self.capacity if self.full else self.idx,
                             (batch_size,), device=self.device)

        batch = {}
        for k, v in self.tensor_dict.items():
            batch[k] = v[idxs]

        return batch

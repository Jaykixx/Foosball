from utilities.models.networks.value_network import ValueNetwork
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses

from torch import optim
from torch import nn
import torch


class WocarWrapper:

    def __init__(self, base_agent, params):
        self.base_agent = base_agent

        # Add the worst case (woc) critic for loss calculation
        build_config = {
            'normalize_input': self.normalize_input,
            'cnn_bypass_input_shape': self.actions_num,
            'input_shape': self.obs_shape,
            'output_shape': 1
        }
        self.woc_critic = ValueNetwork(params['network'], **build_config)
        self.woc_critic.to(self.ppo_device)
        self.woc_critic_coef_target = self.config.get('woc_critic_coef', 0.2)
        self.obs_dist = self.config.get('obs_dist', 0.15)
        self.woc_lr = self.config.get('woc_lr', 1e-5)
        self.woc_optimizer = optim.Adam(
            self.woc_critic.parameters(), self.woc_lr, eps=1e-08, weight_decay=self.weight_decay
        )

    def __getattr__(self, item):
        return getattr(self.base_agent, item)

    @property
    def woc_critic_coef(self):
        start_phase = self.max_epochs * 0.15
        end_phase = self.max_epochs * 0.5
        if self.epoch_num < start_phase:
            return 0
        elif (self.epoch_num >= start_phase) and (self.epoch_num < end_phase):
            return self.woc_critic_coef_target*(self.epoch_num-start_phase) / (end_phase-start_phase)
        else:
            return self.woc_critic_coef_target

    def init_tensors(self):
        self.base_agent.init_tensors()
        self.tensor_list += ['rewards']

    def set_full_state_weights(self, weights, set_epoch=True):
        self.base_agent.set_full_state_weights(weights, set_epoch)
        self.woc_critic.set_state(weights['woc_critic'])
        self.woc_optimizer.load_state_dict(weights['woc_optimizer'])

    def get_full_state_weights(self):
        state = self.base_agent.get_full_state_weights()
        state['woc_critic'] = self.woc_critic.get_state()
        state['woc_optimizer'] = self.woc_optimizer.state_dict()
        return state

    def train_actor_critic(self, input_dict):
        self.train_woc_critic(input_dict)
        self.calc_gradients(input_dict)
        return self.train_result

    def train_woc_critic(self, input_dict):
        obs_batch = input_dict['obs'].clone()
        obs_batch = self._preproc_obs(obs_batch)  # TODO: Ignore low level ASE?
        actions_batch = input_dict['actions'].clone()
        returns = input_dict['woc_returns']

        # compute loss and step
        self.woc_critic.train()
        self.woc_optimizer.zero_grad()
        batch_dict = {
            'obs': obs_batch,
            'cnn_bypass_obs': actions_batch
        }
        woc_qvalues = self.woc_critic(batch_dict)['mus']

        woc_critic_loss = (returns - woc_qvalues)**2
        woc_critic_loss = woc_critic_loss.mean()
        woc_critic_loss.backward()
        self.woc_optimizer.step()
        self.woc_critic.eval()

    def _compute_model_bounds(self, obses):
        with torch.no_grad():
            # TODO: How to best utilize
            # TODO: Now has access to task- and joint-obs indices
            # normed = self.model.norm_obs(obses.clone())
            # lb, ub = normed - self.obs_dist, normed + self.obs_dist
            lb, ub = obses.clone(), obses.clone()
            # lb[:, -10:-7] = lb[:, -10:-7] - self.obs_dist
            # lb[:, -3:] = lb[:, -3:] - self.obs_dist
            # ub[:, -10:-7] = ub[:, -10:-7] + self.obs_dist
            # ub[:, -3:] = ub[:, -3:] + self.obs_dist
            lb[:, -4:] = lb[:, -4:] - self.obs_dist
            ub[:, -4:] = ub[:, -4:] + self.obs_dist
            lb = self.model.norm_obs(lb)
            ub = self.model.norm_obs(ub)
            network = self.model.a2c_network

            if network.has_cnn:
                raise NotImplementedError
            if network.has_rnn:
                raise NotImplementedError

            for layer in network.actor_mlp:
                if isinstance(layer, nn.Linear):
                    mu = layer((ub + lb) / 2).detach()
                    r = (torch.abs(layer.weight.clone().detach()) @ ((ub - lb) / 2).T).T
                    lb, ub = mu - r, mu + r
                else:
                    lb = layer(lb)
                    ub = layer(ub)

            mu = network.mu((ub + lb) / 2).detach()
            r = (torch.abs(network.mu.weight.clone().detach()) @ ((ub - lb) / 2).T).T
            lb, ub = mu - r, mu + r
            lb, ub = network.mu_act(lb).detach(), network.mu_act(ub).detach()

            return lb.detach(), ub.detach()

    def _compute_woc_returns(self, next_obses, rewards, dones):
        # Compute IBP bounds
        self.model.eval()
        lb, ub = self._compute_model_bounds(next_obses)
        self.model.train()

        # Compute worst case actions within bounds
        woc_act = ((ub - lb) / 2 + lb).detach()
        woc_actions = torch.nn.Parameter(woc_act, requires_grad=True)
        woc_opt = optim.Adam([woc_actions], 1e-2, eps=1e-08, weight_decay=self.weight_decay)
        self.woc_critic.eval()
        for _ in range(50):
            woc_opt.zero_grad()
            batch_dict = {
                'obs': next_obses.clone(),
                'cnn_bypass_obs': woc_actions
            }
            loss = self.woc_critic(batch_dict)['mus']
            loss.mean().backward()
            woc_opt.step()
            with torch.no_grad():
                woc_actions.clamp_(lb, ub)

        # compute worst case returns
        batch_dict = {
            'obs': next_obses.clone(),
            'cnn_bypass_obs': woc_actions.detach()
        }
        woc_next_qvalues = self.woc_critic(batch_dict)['mus'].detach()
        return rewards + (1 - dones.unsqueeze(-1)) * self.gamma * woc_next_qvalues

    def prepare_dataset(self, batch_dict):
        self.base_agent.prepare_dataset(batch_dict)

        obses = batch_dict['obses']
        next_obses = batch_dict['next_obses'].clone()
        rewards = batch_dict['rewards']
        dones = batch_dict['dones']
        mus = batch_dict['mus']

        woc_returns = self._compute_woc_returns(next_obses, rewards, dones)
        if self.normalize_value:
            woc_returns = self.value_mean_std(woc_returns)

        # Enhance advantage with worst case q value estimates
        batch_dict = {
            'obs': obses.clone(),
            'cnn_bypass_obs': mus.clone(),
        }
        woc_qvalues = self.woc_critic(batch_dict)['mus'][:, 0].detach()

        dataset_dict = self.dataset.values_dict
        dataset_dict['next_obses'] = next_obses
        dataset_dict['woc_returns'] = woc_returns
        dataset_dict['advantages'] += self.woc_critic_coef * woc_qvalues

        self.dataset.update_values_dict(dataset_dict)

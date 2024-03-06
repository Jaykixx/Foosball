from rl_games.common.a2c_common import dist, swap_and_flatten01
from utilities.models.agents.base_agent import A2CAgent
from utilities.models.custom_network import CustomNetwork
from rl_games.algos_torch import central_value
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
from torch import nn
import torch
import numpy as np
import time


class WocaRA2CAgent(A2CAgent):

    def __init__(self, base_name, params):
        super(WocaRA2CAgent, self).__init__(base_name, params)

        # Add the worst case (woc) critic for loss calculation
        build_config = {
            # TODO: If normalization is added check compute_woc_returns() to not normalize next_obs multiple times
            # 'normalize_input': self.normalize_input,
            'cnn_bypass_input_shape': self.actions_num,
            'input_shape': self.obs_shape,
            'output_shape': 1
        }
        self.woc_critic = CustomNetwork(params['network'], **build_config)
        self.woc_critic.to(self.ppo_device)
        self.woc_critic_coef_target = self.config.get('woc_critic_coef', 0.2)
        self.obs_dist = self.config.get('obs_dist', 0.15)
        self.woc_optimizer = optim.Adam(
            self.woc_critic.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay
        )

    @property
    def woc_critic_coef(self):
        return self.woc_critic_coef_target  #  * self.epoch_num / self.max_epochs

    def init_tensors(self):
        super(WocaRA2CAgent, self).init_tensors()
        self.tensor_list += ['rewards']

    def save(self, fn):
        # TODO: Adjust to accomodate woc_critic
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        # TODO: Adjust to accomodate woc_critic
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def train_actor_critic(self, input_dict):
        self.train_woc_critic(input_dict)
        self.calc_gradients(input_dict)
        return self.train_result

    def train_woc_critic(self, input_dict):
        obs_batch = input_dict['obs'].clone()
        obs_batch = self._preproc_obs(obs_batch)
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

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs'].clone()
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }
        if self.is_ndp:
            dmp_init_obs_batch = input_dict['dmp_init_obs']
            progress_batch = input_dict['progress_buf']
            proc_dmp_init_obs_batch = self._preproc_obs(dmp_init_obs_batch)
            batch_dict['dmp_init_obs'] = proc_dmp_init_obs_batch
            batch_dict['progress_buf'] = progress_batch

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # Enhance advantage with worst case q value estimates
            batch_dict = {'obs': torch.cat((obs_batch.clone(), mu), dim=-1)}
            woc_qvalues = self.woc_critic(batch_dict)['mus'][:, 0].detach()
            woc_adv = advantage + self.woc_critic_coef * woc_qvalues
            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, woc_adv, self.ppo, curr_e_clip
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            e_loss = entropy * self.entropy_coef
            loss = a_loss + 0.5 * c_loss * self.critic_coef - e_loss + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(
            self,
            {
                'values': value_preds_batch,
                'returns': return_batch,
                'new_neglogp': action_log_probs,
                'old_neglogp': old_action_log_probs_batch,
                'masks': rnn_masks},
            curr_e_clip, 0
        )

        self.train_result = (
            a_loss, c_loss, entropy, kl_dist, self.last_lr,
            lr_mul, mu.detach(), sigma.detach(), b_loss
        )

    def _compute_model_bounds(self, obses):

        def compute_affine_bounds(layer, llb, lub):
            mu = layer((lub + llb) / 2)
            r = (torch.abs(layer.weight) @ ((lub - llb) / 2).T).T
            return mu - r, mu + r

        lb, ub = obses - self.obs_dist, obses + self.obs_dist
        network = self.model.a2c_network

        if network.has_cnn:
            raise NotImplementedError
        if network.has_rnn:
            raise NotImplementedError

        for layer in network.actor_mlp:
            if isinstance(layer, nn.Linear):
                lb, ub = compute_affine_bounds(layer, lb, ub)
            else:
                lb = layer(lb)
                ub = layer(ub)

        lb, ub = compute_affine_bounds(network.mu, lb, ub)
        lb, ub = network.mu_act(lb), network.mu_act(ub)

        return lb, ub

    def _compute_woc_returns(self, next_obses, rewards, dones):
        # Compute IBP bounds
        self.model.eval()
        with torch.no_grad():
            normed_next_obs_batch = self.model.norm_obs(next_obses.clone())
            lb, ub = self._compute_model_bounds(normed_next_obs_batch)
            lb = lb.detach()
            ub = ub.detach()
        self.model.train()

        # Compute worst case actions within bounds
        woc_actions = (ub - lb) / 2 + lb
        woc_actions = torch.nn.Parameter(woc_actions, requires_grad=True)
        optimizer = optim.Adam([woc_actions], 1e-2, eps=1e-08, weight_decay=self.weight_decay)
        self.woc_critic.eval()
        for _ in range(50):
            optimizer.zero_grad()
            batch_dict = {
                'obs': next_obses.clone(),
                'cnn_bypass_obs': woc_actions
            }
            loss = self.woc_critic(batch_dict)['mus']
            loss.mean().backward()
            optimizer.step()
            with torch.no_grad():
                woc_actions.clamp_(lb, ub)

        # compute worst case returns
        batch_dict = {
            'obs': next_obses.clone(),
            'cnn_bypass_obs': woc_actions
        }
        woc_next_qvalues = self.woc_critic(batch_dict)['mus'].detach()
        return rewards + (1 - dones.unsqueeze(-1)) * self.gamma * woc_next_qvalues

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        next_obses = batch_dict['next_obses']
        returns = batch_dict['returns']
        rewards = batch_dict['rewards']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        if self.is_ndp:
            dmp_init_obs = batch_dict['dmp_init_obs']
            progress_buf = batch_dict['progress_buf']

        advantages = self._compute_advantages(returns, values, rnn_masks)
        woc_returns = self._compute_woc_returns(next_obses, rewards, dones)

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            woc_returns = self.value_mean_std(woc_returns)
            self.value_mean_std.eval()

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['woc_returns'] = woc_returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['next_obses'] = next_obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        if self.is_ndp:
            dataset_dict['dmp_init_obs'] = dmp_init_obs
            dataset_dict['progress_buf'] = progress_buf

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

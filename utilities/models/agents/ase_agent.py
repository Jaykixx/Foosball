from omniisaacgymenvs.utils.config_utils.path_utils import *
from rl_games.common.a2c_common import *

from utilities.functions.rw_csv_files import read_csv
from utilities.models.networks import DiscEncNetwork
from utilities.models.players import LowLevelPlayer, ase_player
from utilities.models.agents import A2CAgent
from utilities.functions.latent_sampling import *
from utilities.buffers import ReplayBuffer

from abc import abstractmethod
from gym import spaces
import numpy as np
import torch
import os


def ase_agent(base_name, params):
    level = params['train_level']

    if level == 'high':
        return HighLevelAgent(base_name, params)
    else:
        return LowLevelAgent(base_name, params)


class ASEAgent(A2CAgent):

    def __init__(self, base_name, params):
        self.latent_dimension = params['config']["latent_dimension"]

        A2CAgent.__init__(self, base_name, params)

        self.disc_enc = self.load_disc_enc_networks(params)
        self.disc_enc.to(self.ppo_device)

    def load_disc_enc_networks(self, params):
        if hasattr(self, 'ref_motion_type') and self.ref_motion_type == 'dmp':
            n = params['dmp']['rbfs']['numRBFs']
            input_shape = n * int(self.joint_obs_dimension / 2)
        else:
            # obs + next_obs of joints
            input_shape = 2 * self.joint_obs_dimension
        disc_enc_params = params['low_level_params']['disc_enc_network']
        return DiscEncNetwork(disc_enc_params, input_shape=input_shape)

    def prepare_dataset(self, batch_dict):
        A2CAgent.prepare_dataset(self, batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['latents'] = batch_dict.get('latents', None)
        self.dataset.update_values_dict(dataset_dict)


class HighLevelAgent(ASEAgent):

    def __init__(self, base_name, params):
        ASEAgent.__init__(self, base_name, params)

        self.disc_enc.eval()
        self.latent_repetition = self.config["latent_repetition"]

        # Weights
        self.w_task = self.config["w_task"]
        self.w_style = self.config["w_style"]

        self.low_level_player = self.load_low_level_networks(params)

        self.apply_dmps = self.low_level_player.apply_dmps
        if self.apply_dmps:
            self.build_dmp_models(params['low_level_params']['dmp'])

    def _get_policy_out_num(self):
        return self.latent_dimension

    def load_low_level_networks(self, params):
        low_params = params['low_level_params']
        env_info = self.env_info.copy()
        env_info['observation_space'] = env_info['joint_observation_space']
        low_params['config']['env_info'] = env_info
        low_params['config']['vec_env'] = self.vec_env.env
        player = LowLevelPlayer(low_params)

        checkpoint = params['high_level_params']['low_level_checkpoint']
        if checkpoint is not None and checkpoint != '':
            checkpoint = retrieve_checkpoint_path(checkpoint)
            player.restore(checkpoint)
            checkpoint = torch_ext.load_checkpoint(checkpoint)
            self.disc_enc.set_state(checkpoint['disc_enc'])

        if player.is_rnn:
            player.init_rnn()

        player.has_batch_dimension = True
        player.batch_size = self.num_actors * self.num_agents

        return player

    def get_low_level_action(self, obs, latents):
        joint_obs = obs['obs'][:, :self.joint_obs_dimension]
        normed_latents = latents / torch.norm(latents, dim=-1, keepdim=True)
        self.low_level_player.current_latents = normed_latents
        return self.low_level_player.get_action(joint_obs, is_determenistic=True)

    def env_step(self, actions):
        batch_size = self.num_actors * self.num_agents
        task_rewards = torch.zeros((batch_size, 1), device=self.device)
        disc_rewards = torch.zeros((batch_size, 1), device=self.device)
        dones = torch.zeros_like(self.dones)
        infos = {}  # TODO: Check how infos propagate. Now only latest info counts
        for i in range(self.latent_repetition):
            low_level_actions = self.get_low_level_action(self.obs, actions)
            obs, rewards, new_dones, infos = ASEAgent.env_step(self, low_level_actions)

            joint_obs = self.obs['obs'][:, :self.joint_obs_dimension]
            joint_next_obs = obs['obs'][:, :self.joint_obs_dimension]
            transition = torch.cat((joint_obs, joint_next_obs), dim=-1)
            disc_prediction, _ = self.disc_enc(transition)
            # Rewards are only counted until reset
            # TODO: Why are dones in uint8? Check OmniIsaacGymEnvs
            task_rewards[~dones.bool()] += rewards[~dones.bool()]
            disc_rewards[~dones.bool()] += -torch.log(1 - disc_prediction)[~dones.bool()]
            dones = (dones.bool() | new_dones.bool()).to(torch.uint8)

            self.obs['obs'] = obs['obs']
            self.obs['states'] = obs['states']

        rewards = self.w_task * task_rewards + self.w_style * disc_rewards
        return self.obs, rewards, dones, infos


class LowLevelAgent(ASEAgent):

    def __init__(self, base_name, params):
        self.ref_motion_type = params['config']['ref_motion_type']

        ASEAgent.__init__(self, base_name, params)

        # Buffers
        self.real_buffer_size = self.config["real_replay_buffer_size"]
        self.fake_buffer_size = self.config["fake_replay_buffer_size"]
        self.disc_minibatch_size = self.config["disc_minibatch_size"]

        # Latents
        batch_size = self.num_actors * self.num_agents
        self.latents = torch.zeros((batch_size, self.latent_dimension), device=self.device)
        self.latent_steps_min = self.config["latent_steps_min"]
        self.latent_steps_max = self.config["latent_steps_max"]

        # Weights
        self.w_l2_logit_reg = self.config["w_l2_logit_reg"]
        self.w_grad_penalty = self.config['gradient_penalty']
        self.w_diversity = self.config['w_diversity']
        self.w_disc_rew = self.config['w_disc_rew']
        self.w_enc_rew = self.config['w_enc_rew']

        self.ref_motion_path = self.config['ref_motion_path']
        if self.ref_motion_type == 'dmp':
            if not hasattr(self, 'dmp_config'):
                raise Exception('No DMP config was found.')
            numRBFs = self.dmp_config['rbfs']['numRBFs']
            dof = self.actions_num - self.task_has_gripper
            self.num_dmp_params = dof * numRBFs

        self.disc_loss = torch.nn.BCEWithLogitsLoss()
        self.disc_enc_lr = self.config.get('disc_enc_learning_rate', 1e-5)
        self.disc_enc_optimizer = torch.optim.Adam(
            self.disc_enc.parameters(), self.disc_enc_lr, eps=1e-08,  # weight_decay=self.weight_decay
        )

    def _get_policy_input_shape(self):
        if self.ref_motion_type == 'trajectory':
            return (self.obs_shape[0] + self.latent_dimension,)
        else:
            return (self.latent_dimension,)

    def _get_policy_out_num(self):
        if self.ref_motion_type == 'dmp':
            return self.num_dmp_params
        else:
            return ASEAgent._get_policy_out_num(self)

    def get_adjusted_env_info(self):
        """ Used to get right dimensions for experience buffer """
        env_info = self.env_info.copy()
        env_info['action_space'] = spaces.Box(
            np.ones((self._get_policy_out_num(),), dtype=np.float32) * -1.0,
            np.ones((self._get_policy_out_num(),), dtype=np.float32) * 1.0,
        )
        return env_info

    def load_reference_motion_data(self, path):
        if os.path.isdir(path):
            results = {}
            for item in os.listdir(path):
                sub_path = os.path.join(path, item)
                result = self.load_reference_motion_data(sub_path)
                if isinstance(result, dict):
                    for k, v in result.items():
                        if k in results.keys():
                            results[k] = torch.cat((results[k], v), dim=0)
                        else:
                            results[k] = v
                elif torch.is_tensor(result):
                    if isinstance(results, dict):
                        results = result
                    else:
                        results = torch.cat((results, result), dim=0)
        else:
            if path.endswith('.csv'):
                result = read_csv(path, delimiter=',', device=self.device)
                recorded_obs = torch.stack(list(result.values()), dim=-1)
                results = {
                    'obs': recorded_obs[:-1],
                    'next_obs': recorded_obs[1:]
                }
            elif path.endswith('.pth'):
                results = torch.load(path).to(self.device)
            else:
                results = {}

        return results

    def set_full_state_weights(self, weights, set_epoch=True):
        A2CAgent.set_full_state_weights(self, weights, set_epoch)
        self.disc_enc.set_state(weights['disc_enc'])
        self.disc_enc_optimizer.load_state_dict(weights['disc_enc_optimizer'])

    def get_full_state_weights(self):
        state = A2CAgent.get_full_state_weights(self)
        state['disc_enc'] = self.disc_enc.get_state()
        state['disc_enc_optimizer'] = self.disc_enc_optimizer.state_dict()
        return state

    def init_tensors(self):
        ASEAgent.init_tensors(self)
        aux_tensor_dict = {'latents': (self.latent_dimension,)}
        self.experience_buffer._init_from_aux_dict(aux_tensor_dict)

        if self.ref_motion_type == 'trajectory':
            tensor_dict = {'obs': self.obs_shape, 'next_obs': self.obs_shape}
        else:
            tensor_dict = {'dmps': (self.num_dmp_params,)}
        self.real_replay_buffer = ReplayBuffer(tensor_dict, self.real_buffer_size, self.device)

        tensor_dict['latents'] = (self.latent_dimension,)
        self.fake_replay_buffer = ReplayBuffer(tensor_dict, self.fake_buffer_size, self.device)

        path = to_absolute_path(self.ref_motion_path)
        ref_motion = self.load_reference_motion_data(path)
        if not isinstance(ref_motion, dict):
            ref_motion = {'dmps': ref_motion}
        self.real_replay_buffer.add(ref_motion)

        self.update_list += ['latents']
        self.tensor_list += ['latents']

    def env_reset(self):
        self._update_latent_on_step = torch.zeros_like(self._env_progress_buffer)
        self._update_latents()
        return ASEAgent.env_reset(self)

    def _update_latents(self):
        update = self._update_latent_on_step <= self._env_progress_buffer
        num_updates = update.sum()
        if update.any():
            self.latents[update] = sample_easy_latents(
                num_updates, self.latent_dimension, self.device
            )
            self._update_latent_on_step[update] += torch.randint_like(
                self._update_latent_on_step[update],
                low=self.latent_steps_min, high=self.latent_steps_max
            )

    def _preproc_obs(self, obs_batch):
        obs_batch = ASEAgent._preproc_obs(self, obs_batch)
        if self.ref_motion_type == 'trajectory':
            obs_batch = torch.cat((obs_batch, self.latents), dim=-1)
        else:
            obs_batch = self.latents
        return obs_batch

    def get_action_values(self, obs):
        res_dict = ASEAgent.get_action_values(self, obs)
        res_dict['latents'] = self.latents.clone()
        return res_dict

    def set_eval(self):
        self.disc_enc.eval()
        ASEAgent.set_eval(self)

    def set_train(self):
        self.disc_enc.train()
        ASEAgent.set_train(self)

    def train_actor_critic(self, input_dict):
        self.train_disc_enc(input_dict)
        self.calc_gradients(input_dict)
        return self.train_result

    def train_disc_enc(self, input_dict):
        # TODO: Don't rely just on buffer and use truly as replay buffer instead
        self.disc_enc_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            real_dict = self.real_replay_buffer.sample(self.disc_minibatch_size)
            fake_dict = self.fake_replay_buffer.sample(self.minibatch_size)

            if self.ref_motion_type == 'trajectory':
                real_obs = real_dict['obs'].clone()
                real_next_obs = real_dict['next_obs'].clone()
                real_transitions = torch.cat((real_obs, real_next_obs), dim=-1)
                real_transitions = self.disc_enc.get_normalized_input(real_transitions)

                fake_obs = fake_dict['obs'].clone()
                fake_next_obs = fake_dict['next_obs'].clone()
                fake_transitions = torch.cat((fake_obs, fake_next_obs), dim=-1)
                fake_transitions = self.disc_enc.get_normalized_input(fake_transitions)
                latents = fake_dict['latents'].clone()

                disc_loss, disc_info = self._discriminator_loss(real_transitions, fake_transitions)
                enc_loss = self._encoder_loss(fake_transitions, latents)
            else:
                real_input = real_dict['dmps'].clone()
                real_input = self.disc_enc.get_normalized_input(real_input)

                fake_input = fake_dict['dmps'].clone()
                fake_input = self.disc_enc.get_normalized_input(fake_input)
                latents = fake_dict['latents'].clone()

                disc_loss, disc_info = self._discriminator_loss(real_input, fake_input)
                enc_loss = self._encoder_loss(fake_input, latents)

            loss = disc_loss + enc_loss

        loss.backward()
        self.disc_enc_optimizer.step()

        self.train_disc_enc_results['real_data_prediction_loss'].append(
            disc_info['real_loss']
        )
        self.train_disc_enc_results['fake_data_prediction_loss'].append(
            disc_info['fake_loss']
        )
        self.train_disc_enc_results['real_data_prediction_accuracy'].append(
            disc_info['real_acc']
        )
        self.train_disc_enc_results['fake_data_prediction_accuracy'].append(
            disc_info['fake_acc']
        )
        self.train_disc_enc_results['discriminator_loss'].append(disc_loss)
        self.train_disc_enc_results['encoder_loss'].append(enc_loss)

    def _encoder_loss(self, fake_transitions, latents):
        enc_logits = self.disc_enc.get_enc_logits(fake_transitions)
        enc_prediction = enc_logits / enc_logits.norm(dim=-1, keepdim=True)
        enc_err = - torch.sum(enc_prediction * latents, dim=-1, keepdim=True)
        return torch.mean(enc_err)

    def _discriminator_loss(self, real_transitions, fake_transitions):
        # Binary Cross Entropy
        real_transitions = real_transitions.requires_grad_(True)
        real_predictions = self.disc_enc.get_disc_logits(real_transitions)
        fake_predictions = self.disc_enc.get_disc_logits(fake_transitions)

        real_loss = self.disc_loss(real_predictions, torch.ones_like(real_predictions))
        fake_loss = self.disc_loss(fake_predictions, torch.zeros_like(fake_predictions))
        disc_loss = 0.5 * (real_loss + fake_loss)

        # Gradient penalty: set all gradients in points from the reference motion data 0
        disc_demo_grad = torch.autograd.grad(
            outputs=real_predictions,
            inputs=real_transitions,
            grad_outputs=torch.ones_like(real_predictions),
            create_graph=True, retain_graph=True, only_inputs=True
        )

        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_loss += torch.mean(disc_demo_grad) * self.w_grad_penalty

        # Logit regularisation
        logit_weights = self.disc_enc.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self.w_l2_logit_reg * disc_logit_loss

        fake_acc, real_acc = self._compute_disc_acc(
            real_predictions.detach(), fake_predictions.detach()
        )
        result_info = {
            'fake_loss': fake_loss.detach(),
            'real_loss': real_loss.detach(),
            'fake_acc': fake_acc.detach(),
            'real_acc': real_acc.detach()
        }
        return disc_loss, result_info

    def _compute_disc_acc(self, real_predictions, fake_predictions):
        fake_acc = fake_predictions < 0
        fake_acc = torch.mean(fake_acc.float())
        real_acc = real_predictions > 0
        real_acc = torch.mean(real_acc.float())
        return fake_acc, real_acc

    def diversity_loss(self, obs_batch, latents_batch, actions_batch, mu):
        # Compute alternative actions
        new_z = sample_easy_latents(obs_batch.shape[0], self.latent_dimension, self.device)
        if self.ref_motion_type == 'trajectory':
            obs_input = torch.cat((obs_batch, new_z), dim=-1)
        else:
            obs_input = new_z
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_input,
        }
        res_dict = self.model(batch_dict)
        new_mu = res_dict['mus']

        mu_clipped = torch.clamp(mu, -1.0, 1.0)
        new_mu_clipped = torch.clamp(new_mu, -1.0, 1.0)
        a_diff = mu_clipped - new_mu_clipped
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * latents_batch
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff
        div_bonus = a_diff / (z_diff + 1e-5)
        return torch.square(1 - div_bonus).mean()

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions'].clone()
        obs_batch = input_dict['obs'].clone()
        obs_batch = ASEAgent._preproc_obs(self, obs_batch)  # Use base to add latents later
        latents_batch = input_dict['latents'].clone()

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        if self.ref_motion_type == 'trajectory':
            obs_input = torch.cat((obs_batch, latents_batch), dim=-1)
        else:
            obs_input = latents_batch
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_input,
        }

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

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip
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

            div_loss = self.diversity_loss(obs_batch, latents_batch, actions_batch, mu)

            loss = a_loss \
                 + 0.5 * c_loss * self.critic_coef \
                 - entropy * self.entropy_coef \
                 + b_loss * self.bounds_loss_coef \
                 + div_loss * self.w_diversity

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
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

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
        self.train_disc_enc_results['div_loss'] = div_loss

    def disc_enc_reward(self, obs, next_obs):
        with torch.no_grad():
            transitions = torch.cat((obs, next_obs), dim=-1)
            disc_prediction, enc_prediction = self.disc_enc(transitions)

            neg_enc_err = torch.sum(enc_prediction * self.latents, dim=-1, keepdim=True)
            enc_rew = torch.clamp_min(neg_enc_err, 0.0)

            disc_rew = - torch.log(
                torch.maximum(
                    1-disc_prediction,
                    torch.tensor(0.0001, device=self.ppo_device)
                )
            )

        return self.w_disc_rew * disc_rew + self.w_enc_rew * enc_rew

    def disc_enc_reward_dmps(self, params):
        with torch.no_grad():
            disc_prediction, enc_prediction = self.disc_enc(params)

            neg_enc_err = torch.sum(enc_prediction * self.latents, dim=-1, keepdim=True)
            enc_rew = torch.clamp_min(neg_enc_err, 0.0)

            disc_rew = - torch.log(
                torch.maximum(
                    1-disc_prediction,
                    torch.tensor(0.0001, device=self.ppo_device)
                )
            )

        return self.w_disc_rew * disc_rew + self.w_enc_rew * enc_rew

    def env_step(self, actions):
        if self.ref_motion_type == 'trajectory':
            obs, rewards, dones, infos = ASEAgent.env_step(self, actions)
            rewards = self.disc_enc_reward(self.obs["obs"], obs["obs"]).detach()
        else:  # ref_motions given as DMPs
            rewards = self.disc_enc_reward_dmps(actions).detach()
            obs, dones, infos = self.obs, self.dones, {}
        self._update_latents()
        return obs, rewards, dones, infos

    def prepare_dataset(self, batch_dict):
        if self.ref_motion_type == 'trajectory':
            buffer_dict = {
                'obs': batch_dict['obses'].clone(),
                'next_obs': batch_dict['next_obses'].clone(),
                'latents': batch_dict['latents'].clone()
            }
        else:
            buffer_dict = {
                'dmps': batch_dict['actions'].clone(),
                'latents': batch_dict['latents'].clone()
            }
        self.fake_replay_buffer.add(buffer_dict)
        ASEAgent.prepare_dataset(self, batch_dict)

    def train_epoch(self):
        self.train_disc_enc_results = {
            'discriminator_loss': [],
            'encoder_loss': [],
            'real_data_prediction_loss': [],
            'fake_data_prediction_loss': [],
            'real_data_prediction_accuracy': [],
            'fake_data_prediction_accuracy': []
        }
        results = ASEAgent.train_epoch(self)

        frame = self.frame // self.num_agents
        for k, v in self.train_disc_enc_results.items():
            if isinstance(v, list):
                val = torch_ext.mean_list(v).item()
            else:
                val = v.item()
            self.writer.add_scalar('losses/' + k, val, frame)

        return results

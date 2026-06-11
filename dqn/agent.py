import msgpack

from .utils import ABCMeta, abstract_attribute
from .network import Actor_Network, Critic_Network

import os
import time
import math
import random
import numpy as np
from collections import deque
from datetime import timedelta

import torch 
from torch.utils.tensorboard import SummaryWriter

class Agent(metaclass=ABCMeta):
    def __init__(self, actor_lr, critic_lr, gamma, bs, clip_eps, gae_lambda, epochs, entropy_coef, input_dim, output_dim,
                 save_frequency, log_frequency, save_dir, log_dir, load, algo, gpu):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = bs
        self.entropy_coef = entropy_coef

        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.epochs = epochs

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.load = load
        self.info_loss = 0
        # self.e = None # This is epsilon

        self.memory = []
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None


        self.step = 0  # training step
        self.resume_step = 0  # training step from loaded model
        self.event_count = 0
        self.ep_info_buffer = deque([], maxlen=100)

        path = algo + '_actorlr' + str(actor_lr) + '_criticlr' + str(critic_lr)
        self.save_path = save_dir + path + '_' + 'model.pack'
        self.summary_writer = SummaryWriter(log_dir + path + '/')

        self.device = torch.device(("cuda:"+ gpu) if torch.cuda.is_available() else "cpu")
        print("DEVICE", "=", self.device, "" if not torch.cuda.is_available() else torch.cuda.get_device_name(self.device))

        self.start_time = time.time()


    @abstract_attribute
    def actor(self):
        pass

    @abstract_attribute
    def critic(self):
        pass

    def learn(self, next_obs):
        raise NotImplementedError

    def store_transition(self, obs, action, log_prob, value, reward, done, info, train=False):
        self.memory.append((obs, action, log_prob.item(), value.item(), reward, done))
        if train and done:
            self.event_count += 1
            self.ep_info_buffer.append(
                {'st': info['steps'], 'r': info['reward'], 's': info['successes'], 'c': info['collisions'],
                 'f': info["failures"], 'l': self.info_loss})
            print("success", info['successes'], "failure", info["failures"], "collison", info['collisions'])

    def store_transition_test(self, obs, action, reward, done, info):
        if done:
            self.event_count += 1
            self.ep_info_buffer.append(
                {'st': info['steps'], 'r': info['reward'], 's': info['successes'], 'c': info['collisions'],
                 'f': info["failures"], 'l': 0.0})
            print("success", info['successes'], "failure", info["failures"], "collison", info['collisions'])

    def compute_gae(self, next_value):
        rewards, values, dones = [], [], []

        for (_, _, _, v, r, d) in self.memory:
            rewards.append(r)
            values.append(v)
            dones.append(d)

        values = values + [next_value]

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return torch.tensor(returns).to(self.device), torch.tensor(advantages).to(self.device)

    def choose_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)

        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_t)

        return action.item(), log_prob.detach(), value.detach()

    def choose_action_test(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_t)
            probs = torch.softmax(logits, dim=-1)

            action = torch.argmax(probs)

        return action.item()

    def load_model(self):
        """
        Loads the PPO saved weigths to resume training, including filling the ep_info_buffer
        """
        if self.load and os.path.exists(self.save_path):
            print()
            print("Resume training from " + self.save_path + "...")
            with open(self.save_path, 'rb') as f:
                params_dict = msgpack.loads(f.read())

            # load actor
            actor_params = {
                k: torch.as_tensor(np.array(v), device=self.device)
                for k, v in params_dict['actor'].items()
            }
            self.actor.load_state_dict(actor_params)

            # load critic
            critic_params = {
                k: torch.as_tensor(np.array(v), device=self.device)
                for k, v in params_dict['critic'].items()
            }
            self.critic.load_state_dict(critic_params)

            self.resume_step = params_dict['step']
            self.step = self.resume_step
            self.episode_count = params_dict['episode_count']

    def load_model_test(self):
        """
        Load PPO model for evaluation (no training state needed)
        """
        if self.load and os.path.exists(self.save_path):
            print("Loading PPO model for testing...")

            with open(self.save_path, 'rb') as f:
                params_dict = msgpack.loads(f.read())

            # load actor
            actor_params = {
                k: torch.as_tensor(np.array(v), device=self.device)
                for k, v in params_dict['actor'].items()
            }
            self.actor.load_state_dict(actor_params)

            # load critic (optional but recommended)
            critic_params = {
                k: torch.as_tensor(np.array(v), device=self.device)
                for k, v in params_dict['critic'].items()
            }
            self.critic.load_state_dict(critic_params)

            # restore step if needed
            self.step = params_dict.get('step', 0)

    def save_model(self):
        """
        Saves the PPO model's weights with additional infos (reward, loss)
        """
        # save at a giving frequency and make sure that there were additional steps comparing to the last saved model
        if self.step % self.save_frequency == 0 and self.step > self.resume_step:
            print()
            print("Saving model...")

            # First: use existing save() for actor
            params_dict = {
                'actor': {k: v.detach().cpu().numpy() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.detach().cpu().numpy() for k, v in self.critic.state_dict().items()},
                'step': self.step,
                'episode_count': self.episode_count,
                'rew_mean': self.info_mean('r'),
                'len_mean': self.info_mean('l'),
                'suc_mean': self.info_mean('s'),
                'fail_mean': self.info_mean('f'),
                'col_mean': self.info_mean('c')
            }

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            with open(self.save_path, 'wb') as f:
                f.write(msgpack.dumps(params_dict))

            print("OK!")


    def log(self):
        """
        Prints logs and save them in case of train
        """
        if self.step % self.log_frequency == 0 and self.step > self.resume_step:
            step_mean, rew_mean, len_mean, suc_mean, fail_mean, col_mean = self.info_mean('st'), self.info_mean('r'), self.info_mean('l'), self.info_mean('s'), self.info_mean('f'), self.info_mean('c')

            #col_avg = self.info_mean('avg')

            print()
            print('Step: ', self.step)
            print('Avg Rew: ', rew_mean)
            print('Avg Ep Success: ', suc_mean)
            print('Avg Ep Failure: ', fail_mean)
            print('Avg Ep Collision: ', col_mean)
            print('Loss', len_mean)
            print('Events: ', self.event_count)
            print('---', str(timedelta(seconds=round((time.time() - self.start_time), 0))), '---')

            self.summary_writer.add_scalar('AvgSteps', step_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgRew', rew_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgSuc', suc_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgFail', fail_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgCol', col_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('Loss', len_mean, global_step=(self.event_count))
            # self.summary_writer.add_scalar('Epsilon', self.e, global_step=(self.event_count))
            self.summary_writer.add_scalar('Events', self.event_count, global_step=(self.event_count))


    def log_test(self):
        """
        Prints logs and save them in case of test
        """
        if self.step % self.log_frequency == 0 and self.step > self.resume_step:
            step_mean, rew_mean, _, suc_mean, fail_mean, col_mean = self.info_mean('st'), self.info_mean('r'), self.info_mean('l'), self.info_mean('s'), self.info_mean('f'), self.info_mean('c')

            print()
            print('Step: ', self.step)
            print('Avg Rew: ', rew_mean)
            print('Avg Ep Success: ', suc_mean)
            print('Avg Ep Failure: ', fail_mean)
            print('Avg Ep Collision: ', col_mean)
            print('Events: ', self.event_count)
            print('---', str(timedelta(seconds=round((time.time() - self.start_time), 0))), '---')

            self.summary_writer.add_scalar('AvgSteps', step_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgRew', rew_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgSuc', suc_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgFail', fail_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('AvgCol', col_mean, global_step=(self.event_count))
            self.summary_writer.add_scalar('Events', self.event_count, global_step=(self.event_count))

    def info_mean(self, i):
        """
        Return the mean of i over the last episodes (episodes that are still in the buffer)

        Args:
            i: the info to compute the mean on

        Returns:
            The mean of i over the last episodes or 0 if the info does not exists
        """
        i_mean = np.mean([e[i] for e in self.ep_info_buffer])
        return i_mean if not math.isnan(i_mean) else 0.

class PPOAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(PPOAgent, self).__init__(*args, **kwargs)

    def learn(self, next_obs):
        if len(self.memory) == 0:
            return
        # ===== Fast tensor conversion =====
        obs = torch.tensor([m[0] for m in self.memory], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in self.memory]).to(self.device)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float32).to(self.device)
        values = torch.tensor([m[3] for m in self.memory], dtype=torch.float32).to(self.device)
        # ===== Next value =====
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        next_value = self.critic(next_obs_t).detach().item()
        # ===== GAE =====
        returns, advantages = self.compute_gae(next_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===== Mini-batch setup =====
        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        # ===== PPO update =====
        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # ===== Forward =====
                logits = self.actor(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # ===== PPO ratio =====
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # ===== Clipped objective =====
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                # ===== Critic =====
                values_pred = self.critic(batch_obs).view(-1)
                critic_loss = (batch_returns - values_pred).pow(2).mean()

                # ===== Total loss =====
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # ===== Optimize =====
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss.backward()

                # Stability trick
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.info_loss = loss.item()

                # ===== Clear memory =====
        self.memory = []

class PPORLAgent(PPOAgent):
    def __init__(self, *args, **kwargs):
        super(PPORLAgent, self).__init__(*args, **kwargs)

        # Actor: outputs logits for categorical policy
        self.actor = Actor_Network(self.device, self.input_dim, self.output_dim)
        # Critic: outputs scalar V(s)
        self.critic = Critic_Network(self.device, self.input_dim, 1)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)



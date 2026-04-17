import msgpack

from .utils import ABCMeta, abstract_attribute
from .replay_memory import ReplayMemoryNaive, ReplayMemoryPrioritized
from .network import Network, DuelingDeepQNetwork

import os
import time
import math
import random
import numpy as np
from collections import deque
from datetime import timedelta

import torch 
from torch.utils.tensorboard import SummaryWriter

class BaseAgent(metaclass=ABCMeta):
    def __init__(self, lr, gamma, epsilon_start, epsilon_min, epsilon_decay, epsilon_exp_decay, input_dim, output_dim,
                 batch_size, buffer_size, min_buffer_size, update_target_frequency, target_soft_update, target_soft_update_tau,
                 save_frequency, log_frequency, save_dir, log_dir, load, algo, gpu):
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_exp_decay = epsilon_exp_decay
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.update_target_frequency = update_target_frequency
        self.target_soft_update = target_soft_update
        self.target_soft_update_tau = target_soft_update_tau
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.load = load
        self.info_loss = 0
        self.e = None # This is epsilon

        self.step = 0  # training step
        self.resume_step = 0  # training step from loaded model
        self.episode_count = 0
        self.ep_info_buffer = deque([], maxlen=100)

        path = algo + '_lr' + str(lr)
        self.save_path = save_dir + path + '_' + 'model.pack'
        self.summary_writer = SummaryWriter(log_dir + path + '/')

        self.device = torch.device(("cuda:"+ gpu) if torch.cuda.is_available() else "cpu")
        print("DEVICE", "=", self.device, "" if not torch.cuda.is_available() else torch.cuda.get_device_name(self.device))

        self.start_time = time.time()

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
            print('Episodes: ', self.episode_count)
            print('---', str(timedelta(seconds=round((time.time() - self.start_time), 0))), '---')

            self.summary_writer.add_scalar('AvgSteps', step_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgRew', rew_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgSuc', suc_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgFail', fail_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgCol', col_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('Loss', len_mean, global_step=(self.episode_count))
            if self.e is not None: # only for DQN
                self.summary_writer.add_scalar('Epsilon', self.e, global_step=(self.episode_count))
            self.summary_writer.add_scalar('Episodes', self.episode_count, global_step=(self.episode_count))


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
            print('Episodes: ', self.episode_count)
            print('---', str(timedelta(seconds=round((time.time() - self.start_time), 0))), '---')

            self.summary_writer.add_scalar('AvgSteps', step_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgRew', rew_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgSuc', suc_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgFail', fail_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('AvgCol', col_mean, global_step=(self.episode_count))
            self.summary_writer.add_scalar('Episodes', self.episode_count, global_step=(self.episode_count))

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


class PPOBaseAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = []
        self.actor = None
        self.critic = None

    @abstract_attribute
    def actor(self):
        pass

    @abstract_attribute
    def critic(self):
        pass

    def learn(self,next_obs):
        raise NotImplementedError

    def store_transition(self, obs, action, log_prob, value, reward, done, info, train=False):
        self.memory.append((obs, action, log_prob.item(), value.item(), reward, done))
        if train and done:
            self.episode_count += 1
            self.ep_info_buffer.append({'st': info['steps'],'r': info['reward'], 's': info['successes'], 'c': info['collisions'], 'f': info["failures"] ,'l': self.info_loss})
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

################################################################
# BYIN: for all value-based RL
class DQNBaseAgent(BaseAgent):
    @abstract_attribute
    def replay_memory_buffer(self):
        pass

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        raise NotImplementedError

    def transitions_to_tensor(self, transitions):
        """
        Structurs the compounents of transitions in the batch into tensors

        Args:
            transitions: 

        Returns:
            obses_t: A tensor of all observations in the batch
            actions_t: A tensor of all actions in the batch
            rews_t: A tensor of all rewards in the batch
            dones_t: A tensor of all done flags in the batch
            new_obses_t: A tensor of all new observation in the batch
        """
        obses_t = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).to(self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).to(self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).to(self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32).to(self.device)

        return obses_t, actions_t, rews_t, dones_t, new_obses_t

    def store_transition(self, obs, action, rew, done, new_obs, info, train=False):
        """
        Saves the transition in the replay memory and saves episode log
        """
        self.replay_memory_buffer.store_transition(obs, action, rew, done, new_obs)
        if train and done:
            self.episode_count += 1
            self.ep_info_buffer.append({'st': info['steps'],'r': info['reward'], 's': info['successes'], 'c': info['collisions'], 'f': info["failures"] ,'l': self.info_loss})
            print("success", info['successes'], "failure", info["failures"], "collison", info['collisions'])


    def store_transition_test(self, obs, action, rew, done, new_obs, info):
        """
        Saves the transition in the replay memory and saves episode log
        """
        if done:
            self.episode_count += 1
            self.ep_info_buffer.append({'st': info['steps'],'r': info['reward'], 's': info['successes'], 'c': info['collisions'], 'f': info["failures"] ,'l': self.info_loss})
            print("success", info['successes'], "failure", info["failures"], "collison", info['collisions'])

    def epsilon(self):
        """ep_info_buffer
        Returns the new value of epsilon
        """
        if self.epsilon_exp_decay:
            return np.exp(np.interp(self.step, [0, self.epsilon_decay], [np.log(self.epsilon_start), np.log(self.epsilon_min)]))
        else:
            return np.interp(self.step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_min])

    def choose_action(self, obs):
        """
        Chooses the action based on the observation (including the exploration mechanism)
        """
        action = self.online_network.action(obs)
        self.e = self.epsilon()

        # Choose a random action with probability of epsilon
        if random.random() <= self.e:
            action = random.randint(0, self.output_dim - 1)

        return action
    
    def choose_action_test(self, obs):
        """
        Chooses the action based on the observation (no exploration)
        """
        action = self.online_network.action(obs)

        return action

    def update_target_network(self, force=False):
        """
        Updates the weights of the target network based on the weights of the online network

        Args:
            force (bool): if True update the target network immediatly by copying the weights from the online network 
        """
        if (not self.target_soft_update and self.step % (self.update_target_frequency) == 0) or force:
            self.target_network.load_state_dict(self.online_network.state_dict())

        elif self.target_soft_update:
            # update the target network using a linear combination of the weights of the two models
            for target_network_param, online_network_param in zip(self.target_network.parameters(), self.online_network.parameters()):
                target_network_param.data.copy_(
                    (self.target_soft_update_tau) * online_network_param.data +
                    (1. - (self.target_soft_update_tau)) * target_network_param.data
                )

    def load_model(self):
        """
        Loads the saved weigths to resume training, including filling the ep_info_buffer
        """
        if self.load and os.path.exists(self.save_path):
            print()
            print("Resume training from " + self.save_path + "...")
            self.resume_step, self.episode_count, rew_mean, len_mean, suc_mean, fail_mean, col_mean = self.online_network.load(self.save_path)
            [self.ep_info_buffer.append({'r': rew_mean, 'l': len_mean, 's': suc_mean, 'c': col_mean, 'f': fail_mean}) for _ in range(np.min([self.episode_count, self.ep_info_buffer.maxlen]))]
            print("Step: ", self.resume_step, ", Episodes: ", self.episode_count, ", Avg Rew: ", rew_mean, ", Avg Loss: ", len_mean)

            self.update_target_network(force=True)
            self.step = self.resume_step

    def load_model_test(self):
        """
        Loads the saved weigths to resume training, including filling the ep_info_buffer
        """
        if self.load and os.path.exists(self.save_path):
            self.online_network.load_test(self.save_path)
            self.step = self.resume_step

    def save_model(self):
        """
        Saves the model's weights with additional infos (reward, loss)
        """

        # save at a giving frequency and make sure that there were additional steps comparing to the last saved model
        if self.step % self.save_frequency == 0 and self.step > self.resume_step:
            print()
            print("Saving model...")
            self.online_network.save(self.save_path, self.step, self.episode_count, self.info_mean('r'), self.info_mean('l'), self.info_mean('s'), self.info_mean('f'),self.info_mean('c'))
            print("OK!")


class SimpleAgent(DQNBaseAgent):
    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

    @abstract_attribute
    def replay_memory_buffer(self):
        pass

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        """
        Does a learning iteration using a batch sampled from the replayu memory
        """

        # Sample the batch of transitions
        transitions = self.replay_memory_buffer.sample_transitions()
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        # Compute the target q values
        with torch.no_grad():
            target_q_values = self.target_network(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rews_t + (1 - dones_t) * self.gamma * max_target_q_values

        # Compute the predicted q values 
        online_q_values = self.online_network(obses_t)
        action_q_values = torch.gather(input=online_q_values, dim=1, index=actions_t)

        # Compute the loss
        loss = self.online_network.loss(action_q_values, targets).to(self.device)
        self.info_loss = loss.item()

        # Gradient descent
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()


class DoubleAgent(DQNBaseAgent):
    def __init__(self, *args, **kwargs):
        super(DoubleAgent, self).__init__(*args, **kwargs)

    @abstract_attribute
    def replay_memory_buffer(self):
        pass

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        # Compute loss
        transitions = self.replay_memory_buffer.sample_transitions()
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        with torch.no_grad():
            targets_online_q_values = self.online_network(new_obses_t)
            targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

            targets_target_q_values = self.target_network(new_obses_t)
            targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

            targets = rews_t + (1 - dones_t) * self.gamma * targets_selected_q_values

        online_q_values = self.online_network(obses_t)
        action_q_values = torch.gather(input=online_q_values, dim=1, index=actions_t)

        loss = self.online_network.loss(action_q_values, targets).to(self.device)
        self.info_loss = loss.item()
        # Gradient descent
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

class PerDoubleAgent(DQNBaseAgent):
    def __init__(self, *args, **kwargs):
        super(PerDoubleAgent, self).__init__(*args, **kwargs)

    @abstract_attribute
    def replay_memory_buffer(self):
        pass

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        # Compute loss
        is_weights, tree_indices, transitions = self.replay_memory_buffer.sample_transitions(self.step)
        is_weights_t = torch.as_tensor(np.asarray(is_weights), dtype=torch.float32).to(self.device).unsqueeze(-1)
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        with torch.no_grad():
            targets_online_q_values = self.online_network(new_obses_t)
            targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

            targets_target_q_values = self.target_network(new_obses_t)
            targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

            targets = rews_t + (1 - dones_t) * self.gamma * targets_selected_q_values

        online_q_values = self.online_network(obses_t)
        action_q_values = torch.gather(input=online_q_values, dim=1, index=actions_t)

        with torch.no_grad():
            abs_td_errors_np = torch.abs(targets - action_q_values).detach().cpu().numpy()
            self.replay_memory_buffer.update_batch_priorities(tree_indices, abs_td_errors_np)

        loss = torch.mean(is_weights_t * self.online_network.loss(action_q_values, targets)).to(self.device)
        self.info_loss = loss.item()
        with open("loss.txt", "a") as f:
            # Writing data to a file
            f.write(str(loss) + '\n')

        # Gradient descent
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()


class DQNAgent(SimpleAgent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.replay_memory_buffer = ReplayMemoryNaive(self.buffer_size, self.batch_size)

        self.online_network = Network(self.device, self.lr, self.input_dim, self.output_dim)
        self.target_network = Network(self.device, self.lr, self.input_dim, self.output_dim)

        self.update_target_network(force=True)


class DoubleDQNAgent(DoubleAgent):
    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)

        self.replay_memory_buffer = ReplayMemoryNaive(self.buffer_size, self.batch_size)

        self.online_network = Network(self.device, self.lr, self.input_dim, self.output_dim)
        self.target_network = Network(self.device, self.lr, self.input_dim, self.output_dim)

        self.update_target_network(force=True)

class DuelingDoubleDQNAgent(DoubleAgent):
    def __init__(self, *args, **kwargs):
        super(DuelingDoubleDQNAgent, self).__init__(*args, **kwargs)

        self.replay_memory_buffer = ReplayMemoryNaive(self.buffer_size, self.batch_size)

        self.online_network = DuelingDeepQNetwork(self.device, self.lr, self.input_dim, self.output_dim)
        self.target_network = DuelingDeepQNetwork(self.device, self.lr, self.input_dim, self.output_dim)

        self.update_target_network(force=True)

class PerDuelingDoubleDQNAgent(PerDoubleAgent):
    def __init__(self, *args, **kwargs):
        super(PerDuelingDoubleDQNAgent, self).__init__(*args, **kwargs)

        self.replay_memory_buffer = ReplayMemoryPrioritized(self.buffer_size, self.batch_size, self.epsilon_decay)

        self.online_network = DuelingDeepQNetwork(self.device, self.lr, self.input_dim, self.output_dim)
        self.target_network = DuelingDeepQNetwork(self.device, self.lr, self.input_dim, self.output_dim)

        self.update_target_network(force=True)

class PolicyAgent(PPOBaseAgent):
    def __init__(self, *args, **kwargs):
        super(PolicyAgent, self).__init__(*args, **kwargs)

    def learn(self, next_obs):
        obs = torch.tensor([m[0] for m in self.memory], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in self.memory]).to(self.device)
        old_log_probs = torch.tensor([m[2] for m in self.memory], dtype=torch.float32).to(self.device)
        values = torch.tensor([m[3] for m in self.memory], dtype=torch.float32).to(self.device)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        next_value = self.critic(next_obs_t).detach()

        returns, advantages = self.compute_gae(next_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            values_pred = self.critic(obs).squeeze()
            critic_loss = (returns - values_pred).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Optimize
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            loss.backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

            self.info_loss = loss.item()

        self.memory = []

class PPORLAgent(PolicyAgent):
    def __init__(self, *args, clip_eps=0.2, gae_lambda=0.95, epochs=10, **kwargs):
        super(PPORLAgent, self).__init__(*args, **kwargs)

        # Actor: outputs logits for categorical policy
        self.actor = Network(self.device, self.lr, self.input_dim, self.output_dim)

        # Critic: outputs scalar V(s)
        self.critic = Network(self.device, self.lr, self.input_dim, 1)

        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.epochs = epochs

        # On-policy storage
        self.memory = []
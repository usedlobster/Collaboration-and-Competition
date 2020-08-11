"""
Models.py

Implements the network models/algorithm's

"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import utils


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """
        The Actor class
    """
    def __init__(self, config):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(config.env.state_size, config.ACTOR_FC1_SIZE)
        self.fc2 = nn.Linear(config.ACTOR_FC1_SIZE, config.ACTOR_FC2_SIZE)
        self.fc3 = nn.Linear(config.ACTOR_FC2_SIZE, config.env.action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ reset the weights """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ given state return action """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
        The Critic class
    """
    def __init__(self, config):
        super(Critic, self).__init__()
        in_z = (config.env.state_size + config.env.action_size) * config.env.num_agents
        self.fc1 = nn.Linear(in_z, config.CRITIC_FC1_SIZE)
        self.fc2 = nn.Linear(config.CRITIC_FC1_SIZE, config.CRITIC_FC2_SIZE)
        self.fc3 = nn.Linear(config.CRITIC_FC2_SIZE, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ reset the weights """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ given all the agents states , and their actions return the Q-value critic"""
        # create combined state and action vector
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """
        A  single DDPGAgent
    """
    def __init__(self, config):

        self.config = config

        self.actor_local = Actor(config).to(config.device)
        self.actor_target = Actor(config).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)

        self.critic_local = Critic(config).to(config.device)
        self.critic_target = Critic(config).to(config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC)
        # create a Ornstein-Uhlenbeck object for inject noise into action space
        self.noise = utils.OUNoise(config.env.action_size)

    def step(self, idx):
        """ Sample from buffer and learn """
        experiences = self.config.replay.sample(idx)
        self.learn(experiences)

    def act(self, state, add_noise=True):
        """

        Parameters
        ----------
        state   - current state
        add_noise - do we add noise probably yes .

        Returns
        -------
        the action that the current actor_local would do
        with OU noise added and the results clipped to [-1,+1] interval
        """
        state = torch.from_numpy(state).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """ Reset the noise generator """
        self.noise.reset()

    def learn(self, experiences):
        """ update ddpg given the set of experiences """
        states_list, actions_list, rewards, next_states_list, dones = experiences
        next_states_tensor = torch.cat(next_states_list, dim=1).to(self.config.device)
        states_tensor = torch.cat(states_list, dim=1).to(self.config.device)
        actions_tensor = torch.cat(actions_list, dim=1).to(self.config.device)

        next_actions = [self.actor_target(states) for states in states_list]
        next_actions_tensor = torch.cat(next_actions, dim=1).to(self.config.device)
        q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)
        q_targets = rewards + (self.config.GAMMA * q_targets_next * (1 - dones))
        q_expected = self.critic_local(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = [self.actor_local(states) for states in states_list]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(self.config.device)
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """ update target_model with tau proportion of local_model """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def soft_update_all(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)


class MADDPG:
    """
        The maddpg object, controls the two ddpg model
    """
    def __init__(self, config):

        self.config = config
        self.config.replay = utils.ReplayBuffer(config)
        # create a ddpgagent for each player.
        self.config.agents = [DDPGAgent(config) for _ in range(config.env.num_agents)]

    def step(self, obs):
        """

        Add the observation to the replay buffer,
        and call each agents step function.

        update each agents weight.

        Parameters
        ----------
        obs

        Returns
        -------

        """
        self.config.replay.add(obs)
        if len(self.config.replay) > self.config.BATCH_SIZE:
            for i, a in enumerate(self.config.agents):
                a.step(i)

            for a in self.config.agents:
                a.soft_update_all()

    def act(self, states, add_noise=True):
        """ Get each agents actions and combine  """
        actions = np.zeros([self.config.env.num_agents, self.config.env.action_size])
        for index, agent in enumerate(self.config.agents):
            actions[index, :] = agent.act(states[index], add_noise)

        return actions

    def save_weights(self, filename):
        """ save the actor/critic local/target weights to filename_* """
        for index, agent in enumerate(self.config.agents):
            torch.save(agent.actor_local.state_dict(), f'{filename}_actor_{index}.pth')
            torch.save(agent.critic_local.state_dict(), f'{filename}_critic_{index}.pth')

    def load_weights(self, name):
        """ load the actor/critc local/target weights from filename_* """
        for index, agent in enumerate(self.config.agents):
            filename = f'{name}_actor_{index}.pth'
            if os.path.exists(filename):
                agent.actor_local.load_state_dict(torch.load(filename))
            else:
                print(f'\n model not found {filename}\n')
            # probably dont need to load critic - as just for training ?
            filename = f'{name}_critic_{index}.pth'
            if os.path.exists(filename):
                agent.critic_local.load_state_dict(torch.load(filename))
            else:
                print(f'\n model not found {filename}\n')

    def reset(self):
        """ call each agents reset """
        for agent in self.config.agents:
            agent.reset()

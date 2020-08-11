import copy
import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

    def __init__(self, config):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(config.env.state_size, config.ACTOR_FC1_SIZE)
        self.fc2 = nn.Linear(config.ACTOR_FC1_SIZE, config.ACTOR_FC2_SIZE)
        self.fc3 = nn.Linear(config.ACTOR_FC2_SIZE, config.env.action_size)
        self.reset_parameters()


    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):

    def __init__(self, config):
        super(Critic, self).__init__()

        in_z = (config.env.state_size + config.env.action_size) * config.env.num_agents
        self.fc1 = nn.Linear( in_z, config.CRITIC_FC1_SIZE)
        self.fc2 = nn.Linear(config.CRITIC_FC1_SIZE, config.CRITIC_FC2_SIZE)
        self.fc3 = nn.Linear(config.CRITIC_FC2_SIZE, 1 )
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, config):
        self.config = config
        self.memory = deque(maxlen=config.BUFFER_SIZE)  # internal memory (deque)
        self.batch_size = config.BATCH_SIZE

    def add(self, obs):
        self.memory.append(obs)

    def sample(self,idx ):

        batch = random.sample(self.memory, k=self.config.BATCH_SIZE)
        num_a = self.config.env.num_agents
        dev = self.config.device
        if ( idx == 0 ):
            a_order = [0,1]
        else:
            a_order = [1,0]

        b_s = [torch.from_numpy(np.vstack([b['states'][i] for b in batch])).float().to(dev) for i in a_order]
        b_a = [torch.from_numpy(np.vstack([b['actions'][i] for b in batch])).float().to(dev) for i in a_order]
        b_ns= [torch.from_numpy(np.vstack([b['next_states'][i] for b in batch])).float().to(dev) for i in a_order]
        b_r = [torch.from_numpy(np.vstack([b['rewards'][i] for b in batch])).float().to(dev) for i in a_order]
        b_d = [torch.from_numpy(np.vstack([b['dones'][i] for b in batch]).astype(np.uint8)).float().to(dev) for i in a_order]

        return ( b_s , b_a , b_r[idx] , b_ns , b_d[idx] )

    def __len__(self):
        return len( self.memory )


class DDPGAgent():

    def __init__(self, config):

        self.config = config

        self.actor_local = Actor(config).to(config.device)
        self.actor_target = Actor(config).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)


        self.critic_local = Critic(config).to(config.device)
        self.critic_target = Critic(config).to(config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC , weight_decay = 0 )

        # noise processing
        self.noise = OUNoise(config.env.action_size)

    def step(self,idx ):

            experiences = self.config.replay.sample(idx)
            self.learn(experiences)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences ):


        states_list, actions_list, rewards, next_states_list, dones = experiences

        next_states_tensor = torch.cat(next_states_list, dim=1).to(self.config.device)
        states_tensor = torch.cat(states_list, dim=1).to(self.config.device)
        actions_tensor = torch.cat(actions_list, dim=1).to(self.config.device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(states) for states in states_list]
        next_actions_tensor = torch.cat(next_actions, dim=1).to(self.config.device)
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)
        # Compute Q targets for current states
        Q_targets = rewards + (self.config.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        actions_pred = [self.actor_local(states) for states in states_list]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(self.config.device)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 0.5)
        self.actor_optimizer.step()

    def soft_update_all(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_( tau * local_param.data + (1.0 - tau) * target_param.data)


class MADDPG:

    def __init__(self, config):

        self.config = config
        self.config.replay = ReplayBuffer(config)
        self.config.agents = [DDPGAgent(config) for x in range(config.env.num_agents)]

    def step(self, obs):

        self.config.replay.add(obs)
        if len(self.config.replay) > self.config.BATCH_SIZE:
            for i,a in enumerate( self.config.agents ):
                a.step(i)

            for a in self.config.agents:
                a.soft_update_all()

    def act(self, states, add_noise=True):

        actions = np.zeros([self.config.env.num_agents, self.config.env.action_size])
        for index, agent in enumerate(self.config.agents):
            actions[index, :] = agent.act(states[index], add_noise)

        return actions

    def save_weights(self,filename):

        for index, agent in enumerate(self.config.agents):
            torch.save(agent.actor_local.state_dict() , f'{filename}_actor_{index}.pth')
            torch.save(agent.critic_local.state_dict(), f'{filename}_critic_{index}.pth' )
            
    def load_weights(self,name) :

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
        for agent in self.config.agents:
            agent.reset()

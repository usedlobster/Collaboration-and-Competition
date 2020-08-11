import models
import numpy as np
import torch
# import environment helper
from envhelper import UnityEnvHelper


class AgentTrainer:

    def __init__(self, config):

        # we use config amongst all the diffrent classes 
        # its one step better than using  global variable, but still not ideal. 
        # but it works, and there is a lot of stuff
        # that would otherwise need to be passed around

        self.config = config
        # no enivronment set yet 
        self.config.env = None
        # get device we will be using ( hopefully cuda ? )
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __del__(self):
        self.end()  # tidy up

    def start(self, viewer=False, seed=10000):

        torch.manual_seed(seed)
        np.random.seed(seed)
        # create the enivronment ( no_graphics => noviewer )
        self.config.env = UnityEnvHelper(self.config.AGENT_FILE, seed=seed, no_graphics=not viewer)

    def end(self):
        # close the environment
        if self.config.env != None:
            self.config.env.close()

    '''
        train( ) - Train the agents 

		Run each episode and each frame through 
		the MADDPG model class, 

    '''

    def train(self, max_episodes, model_save_filename):

        # create the MADDPG class
        self.scores = []
        # create the MADDPG class where all the train takes place

        model = models.MADDPG(self.config)
        # iterate over each episode 
        for ith in range(1, max_episodes + 1):
            agent_scores = np.zeros(self.config.env.num_agents)
            states = self.config.env.reset(True)
            model.reset()
            for t in range(1000):

                act = model.act(states)
                obs = self.config.env.step(states, act)
                model.step(obs)
                agent_scores += obs['rewards']
                if np.any(obs['dones']):
                    break
                # this is now next frame
                states = obs['next_states']

            # episode score is max of both agent scores
            episode_score = np.max(agent_scores)
            # add to list of all episode scores 
            self.scores.append(episode_score)
            # we use a slice to get the last 100 scores, and take the average.
            moving_avg = np.mean(self.scores[-100:])
            # print out on same line ( or start a new line every 50 - makes judging progress easier ).
            print(f"\rEpisode {ith:5d}{t:5d} , score = {episode_score :8.3f} average(100) = {moving_avg:8.3f}",
                  end='' if (ith % 50) else '\n')

            # have we finished yet ?
            if moving_avg >= 0.5 and ith >= 100:
                # goal achieved - so save weights
                print(f"\nEnvironment Solved From {(ith - 100):d} episode ")
                model.save_weights(model_save_filename)
                break

    '''
    play( ) - Replay episodes , and record scores per episode.
    '''

    def play(self, max_episodes, model_save_filename, train_mode=False):

        self.scores = []

        model = models.MADDPG(self.config)
        model.load_weights(model_save_filename)

        for ith in range(1, max_episodes + 1):

            agent_scores = np.zeros(self.config.env.num_agents)
            states = self.config.env.reset(train_mode)
            model.reset()
            for t in range(1000):

                act = model.act(states)
                obs = self.config.env.step(states, act)
                agent_scores += obs['rewards']
                if np.any(obs['dones']):
                    break
                states = obs['next_states']

            # episode score is max of both agent scores
            episode_score = np.max(agent_scores)
            self.scores.append(episode_score)

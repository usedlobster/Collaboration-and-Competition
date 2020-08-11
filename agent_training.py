import numpy as np
import torch
import models
from envhelper import UnityEnvHelper

class AgentTrainer:

    def __init__(self, config):

        self.config = config
        self.config.env = None
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __del__(self):
        self.end()

    def start(self, viewer=False , seed = 10000  ):

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.config.env = UnityEnvHelper(self.config.AGENT_FILE, seed=seed, no_graphics=not viewer)

    def end(self):

        if self.config.env != None:
            self.config.env.close()

    def train(self, max_episodes, model_save_filename):

        # create the MADDPG class
        self.scores = []
        model = models.MADDPG(self.config)

        for ith in range(1, max_episodes + 1):
            agent_scores = np.zeros( self.config.env.num_agents )
            states = self.config.env.reset(True)
            model.reset()
            for t in range(1000):

                act = model.act( states  )
                obs = self.config.env.step( states, act )
                model.step( obs )
                agent_scores += obs[ 'rewards' ]
                if np.any( obs['dones'] ):
                    break

                states = obs[ 'next_states']

            # episode score is max of both agent scores
            episode_score = np.max( agent_scores )
            self.scores.append( episode_score )

            moving_avg = np.mean( self.scores[-100:] )

            print(f"\rEpisode {ith:5d}{t:5d} , score = {episode_score :8.3f} average(100) = {moving_avg:8.3f}", end='' if (ith % 50) else '\n')


            if moving_avg >=0.5 and ith >=100 :
                # goal achieved
                print(f"\nEnvironment Solved From {(ith - 100):d} episode ")
                model.save_weights( model_save_filename)
                break


      
    def play(self, max_episodes, model_save_filename):

        self.scores = []

        model = models.MADDPG(self.config)
        model.load_weights( model_save_filename)

        for ith in range(1, max_episodes + 1):

            agent_scores = np.zeros( self.config.env.num_agents )
            states = self.config.env.reset(False)
            model.reset()
            for t in range(1000):

                act = model.act( states  )
                obs = self.config.env.step( states, act )
                agent_scores += obs[ 'rewards' ]
                if np.any( obs['dones'] ):
                    break
                states = obs[ 'next_states']

            # episode score is max of both agent scores
            episode_score = np.max( agent_scores )
            self.scores.append( episode_score )





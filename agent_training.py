import models
import numpy as np
import torch
from envhelper import UnityEnvHelper  # import environment helper


class AgentTrainer:
    """
    AgentTrainer class

    Methods
    -------

    start() -  open the environment
    end() - close the environment
    train() - train the agent
    play() - run a trained agent


    """
    def __init__(self, config):

        # we use config amongst all the different classes
        # its one step better than using  global variable, but still not ideal. 
        # but it works, and there is a lot of stuff
        # that would otherwise need to be passed around

        self.config = config
        # no environment set yet
        self.config.env = None
        # get device we will be using ( hopefully cuda ? )
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scores = []

    def __del__(self):
        self.end()  # tidy up

    def start(self, viewer=False, seed=10000):
        """

        Parameters
        ----------
        viewer  -   if False ( default ) don't show unity window
        seed    -   random seed to set

        Returns
        -------
        None

        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        # create a unity enviornment helper class
        self.config.env = UnityEnvHelper(self.config.AGENT_FILE, seed=seed, no_graphics=not viewer)

    def end(self):
        """ close the unity enivronment helper if one exists """
        # close the environment
        if self.config.env is not None:
            self.config.env.close()

    def train(self, max_episodes, model_save_filename):
        """

        Trains the  MADDPG agent , if the goal is reached save the weights


        Parameters
        ----------
        max_episodes  - maximum episodes to attempt
        model_save_filename - model weight base filename to save weights under

        Returns
        -------
        A list of episode scores

        """
        # reset score list
        self.scores = []
        # create the MADDPG class to train.
        model = models.MADDPG(self.config)
        t = 0
        # iterate over each possible episode
        for ith in range(1, max_episodes + 1):
            agent_scores = np.zeros(self.config.env.num_agents)
            states = self.config.env.reset(True)
            model.reset()
            for t in range(1000):
                # get all the actions for each agent from model
                act = model.act(states)
                # send to environment and get observations back
                obs = self.config.env.step(states, act)
                # send observaetion to model for training
                model.step(obs)
                # update each agents score with rewards
                agent_scores += obs['rewards']
                # check if not done yet
                if np.any(obs['dones']):
                    break
                # update state information with current state
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

        return self.scores

    def play(self, max_episodes, model_load_filename, train_mode=False):
        """

        Play max_episodes using the weights loaded from model_load_Filename

        Parameters
        ----------
        max_episodes  - episodes to play
        model_load_filename - base file name to load model weights from
        train_mode - if True ( max speed of simulator ) otherwise realtime


        Returns
        -------
        list of scores obtained

        """
        self.scores = []

        model = models.MADDPG(self.config)
        model.load_weights(model_load_filename)

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

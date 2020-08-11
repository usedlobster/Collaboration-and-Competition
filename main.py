import matplotlib.pyplot as plt
import numpy as np
from agent_training import AgentTrainer


class ConfigParams(object):
    AGENT_FILE = './Tennis_Linux/Tennis.x86_64'
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.1
    LR_ACTOR = 0.0001
    LR_CRITIC = 0.0001
    #
    ACTOR_FC1_SIZE = 128
    ACTOR_FC2_SIZE = 128
    #
    CRITIC_FC1_SIZE = 128
    CRITIC_FC2_SIZE = 128


def train():
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=False, seed=12345)  # start environment without viewerx
    agent.train(10000, 'model')

    return agent.scores


def validate():
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=True, seed=54321)  # start environment without viewerx
    agent.play(100, 'model', train_mode=True)

    return agent.scores


def play():
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=True, seed=10000)  # start environment without viewerx
    agent.play(1, 'model')

    return agent.scores


scores = play()

"""
    Collaboration and Competition
    ------
    Tennis Environment

"""

# import the AgentTrainer class
from agent_training import AgentTrainer


class ConfigParams(object):
    """ Config Parameter Class """
    AGENT_FILE = './Tennis_Linux/Tennis.x86_64'
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.01
    LR_ACTOR = 0.0002
    LR_CRITIC = 0.0002
    ACTOR_FC1_SIZE = 128
    ACTOR_FC2_SIZE = 128
    CRITIC_FC1_SIZE = 128
    CRITIC_FC2_SIZE = 128


def train():
    """ start training """
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=False, seed=12345)
    agent.train(10000, 'model')
    return agent.scores


def validate():
    """ run a validation cycle """
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=True, seed=54321)
    agent.play(100, 'model', train_mode=True)
    return agent.scores


def play():
    """ play a game """
    agent = AgentTrainer(ConfigParams)
    agent.start(viewer=True, seed=10000)
    agent.play(1, 'model')
    return agent.scores


scores = train()
# scores = validate()
# scores = play()

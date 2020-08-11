# Udacity  Deep Reinforcement Learning

## Project 3 - Collaboration and Competition

### Tennis Enivronment

[![Alt text](https://img.youtube.com/vi/BIa1UGzsWWc/0.jpg)](https://www.youtube.com/watch?v=BIa1UGzsWWc&loop=1)

### Project Details 

For this project, two agents need to be trained to play tennis ( on a 2d court ). Each agent has control of the racket and can move either forward or away from the net, or jumping.

The world is  provided as a virtual environment using Unity Machine Learning Agents ( https://github.com/Unity-Technologies/ml-agents).

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket, stacked over 3 observations given a total of 24 inputs. Each agent receives its own, local observation. 

Two action variables are available, corresponding to movement toward (or away from) the net, and jumping. These actions are in the [-1,+1] interval.

The score is calculated as follows, if an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



The agent code is required to be written in  Python 3 and use the Pytorch framework.. It interacts with a custom provided unity app, that has an interface using the open-source Unity plugin (ML-Agents). For this project it is using ML-Agents version 0.4.0.

### Getting Started

1.  Clone or (download and unzip)  this repository.

    ```bash
    $ git clone https://github.com/usedlobster/Collaboration-and-Competition.git
    ```

    ​

2.  This project requires certain, library dependencies to be consistent - in particular the use of python >=3.6 , and a specific version of the ml_agents library version 0.4.0. These dependencies can be found in the pip requirements.txt file. 

    Different systems will vary, but an example configuration for setting up a conda environment  can be made as follows:-

    ```bash
    # create a conda enviroment 
    $ conda create --name drlnd python=3.6
    $ conda activate drlnd
    # install python dependencies 
    $ pip -r requirements.txt
    # install jupyter notebook kernel 
    $ python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    For more details see [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)

3. Download the correct environment for your operating system.

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (*To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the \**Linux** operating system above.*)

4. Unzip (or decompress) the file, into the same folder this repository has been saved.

### Instructions

The main code functions are found in the following files , agent_training.py  , models.py, utils.py and envhelper.py.

For convenince the interface to these is contained in the *Trainer.ipynb* jupyter/ipython notebook, where one can experemint with the Hyperparameters and record and visualize the results. Also included main.py which you can also edit and just change the last line to run either train(), validate() or play() . 

You need to run code in  ***section 0*** first, but can then run any of sections 1,2 and 3. The unity agents are a bit buggy inside a notebook, so you may have to restart the jupyter kernel after each section.

**0 . Setup **

Firstly, update the variable AGENT_FILE to match the path where you have downloaded the the binary unity agent code, for your particular OS.

If desired to try new hyperparameters ( some arent technically hyper ) , just change them in the 

ConfigParams class object.

And then  run the cells in either of these sections ( the model weights files need to be present if not training again ). 

**1. Training** 

Training can be done  by simply executing the code in this section. The following parameters can be adjusted.

A plot of scores obtained during training is produced. And if a successful solution is found the model weights are saved , with the names `(model)_actor_(N).pth` and `(model)_critic_(N).pth` where N is the agent number, and (model) the f. The critic weights are not strictly needed after training. 

**2. Validation** 

Just to prove the model is indeed solved, we can load the agent again ( with different seed ) , and run another 100 episodes, and visualize the results.

This is usefull todo as its possible the model may have worsened during the later stages of training. 

**3. Play**

For completness ( and visualing the end result), it is possible to play a single episode with the trained model weights. By default with the viewer window and at normal speed - but can be changed. It will print out the final score at the end of each run.



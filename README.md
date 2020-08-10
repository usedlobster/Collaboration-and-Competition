# Udacity  Deep Reinforcement Learning

## Project 2 - Continuous Control
### Reacher Enivronment

[![Alt text](https://img.youtube.com/vi/34Gu98Q33G4/0.jpg)](https://www.youtube.com/watch?v=34Gu98Q33G4&loop=1)

### Project Details 

For this project, we have been given the task of training an agent to conrol twenty double jointed robot arms, to main contact with a moving target. The goal is to move each arm to maintain contact with the moving ( and sometimes stationary ) object for as long as possible. A reward of +0.1 is given for each step that the arms hand ( feeler ) is touching the object.

The world is  provided as a virtual environment using Unity Machine Learning Agents ( https://github.com/Unity-Technologies/ml-agents).

For each arm, the environment provides an observation of the current state  , and returns  an action vector., and reward. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between  -1 and 1.  This is repeated for evey arm so there are 20x33 states , 20x4 actions and 20x1 rewards, and 20x1 done flags ( we assume for this project if one agent is done , they will all done ).

The environment is considered solved , when the average episode score over the last 100 episodes is at least 30. For each episode , the sum of rewards ( observed without discounting ) for each arm is accumulated , the episode score is the mean over all these.

The agent code is required to be written in  Python 3 and use the Pytorch framework.. It interacts with a custom provided unity app, that has an interface using the open-source Unity plugin (ML-Agents). For this project it is using ML-Agents version 0.4.0.

### Getting Started

1.  Clone or (download and unzip)  this repository.

2.  This project requires certain, library dependencies to be consistent - in particular the use of python >=3.6 , and a specific version of the ml_agents library version 0.4.0. These dependencies can be found in the pip requirements.txt file. 

    Different systems will vary, but an example configuration for setting up a conda environment  can be made as follows:-

    ```
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

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (*To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the \**Linux** operating system above.*)

4. Unzip (or decompress) the file, into the same folder this repository has been saved.

### Instructions

The training code is found in the *agent.py* file , along with helper functions envhelper.py and models.py. 

For convenince the interface to these is contained in the *Trainer.ipynb* jupyter/ipython notebook, where one can experemint with the Hyperparameters and record and visualize the results.

You need to run code in  ***section 0*** first! ,  but can then run any of sections 1,2 and 3. The unity agents are a bit buggy inside anotebook, so you may have to restart the kernel after each section.

**0 . Setup **

Firstly, update the variable AGENT_FILE to match the path where you have downloaded the the binary unity agent code, for your OS.

If desired to try new hyperparameters, just change them in the CONFIG_PARAMS class object.

And run the cells in this section.

**1. Training** 

Training can be done  by simply executing the code in this section. The following parameters can be adjusted.

A plot of scores obtained during training is produced. And if a successful solution is found the model weights are save id the model.pth file.

**2. Validation** 

Just to prove the model, we can load the agent again ( with different seed ) , and run another 100 episodes, and visualize the results.

This is usefull todo as its possible the model may have worsened during the later stages of training. ( see my Report.pdf - for more information).

**3. Play**

For completness, it is possible to play a single episode with the trained model weights. By default with the viewer window and at normal speed - but can be changed. It will print out the final score at the end of each run.



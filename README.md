# Implementation of DQN

  This repository contains a minimal DQN code.

# Installation

  1. Download Anaconda and install anaconda
  2. `conda create -n dqn python=3.6`
  3. `source activate dqn`
  4. `pip install tensorflow-gpu`
  5. `conda install -c menpo opencv`
  6. `git clone https://github.com/openai/gym.git`
  7. `cd gym`
  8. `pip install -e .`
  9. `cd ..`
  10. `git clone https://github.com/Nat-D/DQN-Tensorflow`

## To run
  1. open two terminals
  2. run `cd DQN-Tensorflow; source activate dqn` on both windows
  3. run `python dqn.py`
  4. run `tensorboard --logdir experiments/`
  5. open browser and direct to `localhost:6666`

# Reference
  The code is based on implementations by
  - [OpenAI's baseline] (https://github.com/openai/baselines)
  - [DennyBritz's DQN] (https://github.com/dennybritz/reinforcement-learning)

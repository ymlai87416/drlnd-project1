[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

[image3]: images/FixedQ.png "FixQ"

[image4]: images/Result.png "Result"

# Project 1: Navigation 

## Introduction

For this project, An agent is trained to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

Trained Agent

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

```
States look like: [0.         1.         0.         0.         0.27946243 0.
 1.         0.         0.         0.74556106 0.         0.
 1.         0.         0.48277503 0.         1.         0.
 0.         0.30341193 0.         0.         1.         0.
 0.46494457 0.         0.         1.         0.         0.08466676
 0.         1.         0.         0.         0.95967758 0.
 0.        ]
States have length: 37
```

## Algorithm

The algorithm used is Deep Q-Network [1], and here is technical implementation

A list of graph here and also blah blah blah


In file `dqn-agent.py`, there are 2 classes.

Class `Agent`implements the Deep DQN[], which makes use of Experience Replay and Fixed Q-Targets 

Class `ReplayBuffer` store each step agent experienced. The data structure of the record is (state, action, reward, next_state)

For every 4 steps, random sampling from this buffer and pass it to a neural network to learn a better policy.

### Experience Replay

The buffer is set to a size of 100000, each time calling `sample()` returns a batch size of 64 records for neural network training.

### Fixed Q-Targets 

The agent setup 2 neural network, one called local and one called target. All the neural network training is done in local, while target provides a stable TD target for optimizer to work.

![Fixed Q Equation][image3]


File `model.py` implements the neural network used.

The model consist of 3 hidden layers:

1st layer: Fully connected - 37 state variables and 64 outputs

2nd layer: Fully connected - 64 inputs and 64 outputs

3rd layer: Fully connected - 64 inputs to 4 actions


## Result

After training for 1000 episodes, the average score for 100 episode is around 15.

![Reward plot][image4]


## Future works

Here is some idea to further improve the performance and the learning efficiency of the agent.

* Double DQN [2]
* Prioritized Experience Replay [3]
* Dueling DQN [4]
* Rainbow [5]
* Training agent directly using pixels

## Reference

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

[2] Thrun, Sebastian, and Anton Schwartz. "Issues in using function approximation for reinforcement learning." Proceedings of the 1993 Connectionist Models Summer School Hillsdale, NJ. Lawrence Erlbaum. 1993.

[3] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

[4] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).

[5] Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
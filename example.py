"""
A simple example script to simulate a set of homogeneous bandit arms.
"""

import matplotlib.pyplot as plt
import numpy as np

from rmabp import *


def alpha(t):  # Q learning rate
    return 1.0 / np.log(t + np.e)


def p_explore(t):  # exploration rate
    return 0.1


I = 100  # number of bandit arms
L = 5  # number of states available to each arm
discount = 0.99  # discount factor
relative = True  # relative or absolute values
K = int(0.1 * I)  # number of active arms
T = 400  # number of time steps

np.random.seed(278901)

# a multi-armed bandit contains a list of single bandit arms
rmab = RMAB([RestartBandit(L) for i in range(I)])

# This controller assumes the bandit arms are homogenous
controller = homoRMABController(
    rmab,
    QWICladder,
    K,
    relative=relative,
    alpha=alpha,
    RMABargs={"p_explore": p_explore},
)

rmab.randomise_states(steady_dist=True)
controller.reset()
episode = RMABEpisode(rmab.states(), T, rmab.identifiers())
wiepisode = WhittleIndexEpisode(controller, T)
WIsim(rmab, controller, T, episode, wiepisode)

episode.plot_total_reward()
plt.show()
wiepisode.plot_wi(true_wi=rmab.get_wi())
plt.show()

import numpy as np
import time
from collections.abc import Iterable
from gym import spaces, Env

class RandomPolicy(object):
    def __init__(self, env):
        self.num_products = env.num_products

    def get_action_vec(self, states):
        n = states.shape[0]
        actions = []
        for i in range(n):
            actions.append(np.random.choice(self.num_products,2,replace=False))
        return actions

    def get_action(self, state):
        action = np.random.choice(self.num_products,2,replace=False)
        return action

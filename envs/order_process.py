import math
import numpy as np
import time
import logger

class StaticOrderProcess(object):
    '''
    Each order for each product is a bernoulli trial with p equals to dist_param
    '''
    def __init__(self, num_products, dist_param= None, seed = 142):
        self.order_random = np.random.RandomState()
        self.num_products = num_products
        self.set_seed(seed)
        self.age = 0
        if dist_param:
            assert num_products == len(dist_param), 'storage_shape should be consistent with dist_param length'
            self.init_dist_param = np.array(dist_param)
        else:
            self.init_dist_param = np.linspace(0.1,0.9,self.num_products)
        self.dist_param = self.init_dist_param
        self.dynamic_order = False

    def set_seed(self, seed=None):
        if seed:
            self.seed_num = seed
        self.order_random.seed(self.seed_num)

    def reset(self):
        self.set_seed()
        self.age = 0
        self.dist_param = self.init_dist_param

    def get_orders(self, num_envs=1):
        order = self.order_random.binomial(1, self.dist_param)
        self.age += 1
        if num_envs != 1:
            order = np.repeat(order, num_envs).reshape((self.num_products, num_envs)).T
        return order

class SeasonalOrderProcess(object):
    '''
    Each order for each product is a bernoulli trial with p equals to dist_param.
    But change with season.
    '''
    def __init__(self, num_products, dist_param= None, season_length= 500, beta=0.8, rho=0.99, seed = 142):
        self.order_random = np.random.RandomState()
        self.num_products = num_products
        self.set_seed(seed)
        self.num_distinct_season = 2
        self.season_length = season_length

        if dist_param:
            self.init_dist_param = np.array(dist_param)
        else:
            self.init_dist_param = np.linspace(0.1,0.9,self.num_products)

        self.long_term_2p = np.array([self.init_dist_param,
                                     self.init_dist_param[::-1]])*2
        self.dist_param = self.init_dist_param

        self.dynamic_order = True
        self.age = 0
        self.beta = beta
        self.rho = rho

    def set_seed(self, seed=None):
        if seed:
            self.seed_num = seed
        self.order_random.seed(self.seed_num)

    def reset(self):
        self.set_seed()
        self.dist_param = self.init_dist_param
        self.age = 0

    def get_orders(self, num_envs=1):
        self.season = (self.age // self.season_length) % self.num_distinct_season
        self.dist_param = self.beta * self.long_term_2p[self.season]/2 + (1-self.beta) * (self.rho * self.dist_param + \
            (1 - self.rho) * self.long_term_2p[self.season] * \
            self.order_random.rand(self.num_products,))
        self.age += 1
        # print(f'Dynamic p: {self.dist_param}')
        order = self.order_random.binomial(1, self.dist_param)
        if num_envs != 1:
            order = np.repeat(order, num_envs).reshape((self.num_products, num_envs)).T
        return order


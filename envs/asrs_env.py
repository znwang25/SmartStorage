import math
import gym
import matplotlib
from gym import spaces
from gym.utils import seeding
import numpy as np

class ASRSEnv(gym.Env):

    """
    Description:
    There is a storage warehouse with M= W * H bins. M (or W * H) types of products will be store in this warehouse. Each period, there will be an array of orders coming in. 
    A robot can exchange the positions of two bins. The goal is to find the optimal storage plan to make best fullfil the orders.

    Observation:
        Current Storage Map    np.array()   (M,)     
                row number indicates the bin number, the value indicates the the good number being stored in the bin.
                Good number starts from 1.
        Current Period Order   np.array()   (M,)
    State:
        Current Storage Map    np.array()   (M,)     
        Time to receive next order  np.array()   (M,)   
        Current 

    Actions:
        Num	of Action: M Choose 2 + 1
        e.g  (a , b) switch bin a with bin b 
             or do nothing

    Reward:
        Reward is -1 for every step taken

    Starting State:
        A random permutation.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, storage_shape, dist_param = None):
        self.seed()
        self.storage_shape = storage_shape
        self.num_products = np.array(storage_shape).prod()
        self.storage_map = np.random.permutation(self.num_products) + 1
        self.dist_origin_to_exit = 1 # Distance from (0,0,0) to exit
        self._storage_maps = None
        self._num_envs = None
        
        if dist_param == None: 
            self.dist_param = np.array([0.05]*self.num_products)
        else:
            self.dist_param = np.array(dist_param)
        self.max_distance = (np.array(storage_shape)-1).sum()+self.dist_origin_to_exit

        self.viewer = None
        self.steps_beyond_done = None
        self.vectorized = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_bin_coordinate(self,bin_id):
        '''
        Given a bin number, this gives the location of the bin. This can be useful to to calculate distance between bin to exit.
        '''
        a,b,c = self.storage_shape
        x, y, z = bin_id//(b*c),  bin_id%(b*c)//c, bin_id%c
        return x, y, z
    
    def get_distance_to_exit(self,bin_id = None):
        if bin_id is None:
            return np.vstack(self.get_bin_coordinate(np.arange(self.num_products))).sum(axis=0)+self.dist_origin_to_exit
        else:
            return sum(self.get_bin_coordinate(bin_id)) + self.dist_origin_to_exit
    

    def set_storage_map(self, storage_map):
        self.storage_map = storage_map

    def reset(self):
        self._storage_maps = None
        self.storage_map = np.random.permutation(self.num_products)
        self.steps_beyond_done = None
        return np.array(self.storage_map)

    def get_orders(self, num_envs=1):
        if num_envs == 1:
            order = (np.random.uniform(size=self.num_products) < self.dist_param).astype(int)
        else:
            order = (np.random.uniform(size=(num_envs,self.num_products)) < self.dist_param).astype(int)
        return order

    def step(self, action=None):
        '''
        Action should be a tuple (x, y), which indicates that bin number x and bin number y should switch good.
        '''
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        storage_map = self.storage_map
        cost = 0
        order = self.get_orders()
        if (action is not None) and (action[1]!=action[0]):
            storage_map[action[0]], storage_map[action[1]] = storage_map[action[1]], storage_map[action[0]]
            cost +=np.abs(np.array(self.get_bin_coordinate(action[0]))-np.array(self.get_bin_coordinate(action[1]))).sum()
        self.storage_map = storage_map
        return np.array(self.storage_map).copy(), order, cost

    def vec_step(self, actions):
        # assert list(map(self.action_space.contains, actions)).all()
        assert self._storage_maps is not None
        storage_maps = self._storage_maps
        orders = self.get_orders(num_envs=self._num_envs)
        rewards = np.zeros((self._num_envs,))
        for i, action in enumerate(actions):
            if (action is not None) and (action[1]!=action[0]):
                self._storage_maps[i,action[0]], self._storage_maps[i, action[1]] = storage_maps[i, action[1]], storage_maps[i, action[0]]
                rewards[i] += -1
        return np.array(self._storage_maps).copy(),orders, rewards

    def vec_reset(self, num_envs=None):
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._storage_maps = np.vstack(list(map(np.random.permutation,[self.num_products]*num_envs)))
        return np.array(self._storage_maps).copy()


    def vec_set_storage_map(self, storage_maps):
        self._num_envs = len(storage_maps)
        self._storage_maps = storage_maps.copy()

    def render(self, mode='human', iteration=None):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

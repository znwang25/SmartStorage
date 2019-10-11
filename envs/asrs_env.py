﻿import math
import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gym.utils import seeding
import numpy as np
import time
from PIL import ImageDraw,Image,ImageFont
import logger


class ASRSEnv(gym.Env):

    """
    Description:
    There is a storage warehouse with M= W * H bins. M (or W * H) types of products will be store in this warehouse. Each period, there will be an array of orders coming in. 
    A robot can exchange the positions of two bins. The goal is to find the optimal storage plan to make best fullfil the orders.

    Observation:
        Current Storage Map    np.array()   (M,) any 1d array    
                row number indicates the bin number, the value indicates the the good number being stored in the bin.
                Good number starts from 1.
        Current Period Order   np.array()   (M,)
    State:
        Current Storage Map    np.array()   (M,)     
        Time to receive next order  np.array()   (M,)   
        Current 

    Action:
        Num	of Action: M Choose 2 + 1
        e.g  (a , b) switch bin a with bin b 
             or do nothing

    Reward:
        Reward is -1 for every step taken

    Starting State:
        A random permutation.
    
    Parameter type:
        storage_shape: tuple with 1 to 3 int  
        dist_param: list with M numbers in [0, 1] 
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, storage_shape, dist_param = None, seed=42):
        self.seed(seed)
        assert len(storage_shape) <= 3, "storage_shape length should be <= 3"
        self.storage_shape = storage_shape
        self.num_products = np.array(storage_shape).prod()

        self.num_actions = int(self.num_products * (self.num_products - 1) / 2 + 1)
        self.action_dim = 2

        self.reset()
        self.dist_origin_to_exit = 1 # Distance from (0,0,0) to exit
        self._storage_maps = None
        self._num_envs = None

        if dist_param == None: 
            self.dist_param = np.array([0.05]*self.num_products)
        else:
            self.dist_param = np.array(dist_param)
        self.max_distance = (np.array(storage_shape)-1).sum()+self.dist_origin_to_exit

        self._fig = None
        # self.cmap = matplotlib.cm.get_cmap('Spectral')
        self.cmap = matplotlib.cm.get_cmap('coolwarm')
        self.dt = .2
        self._scale = 16
        self.vectorized = True
        self.__name__ = 'ASRSEnv'

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def reset(self):
        self._storage_maps = None
        self.storage_map = np.random.permutation(self.num_products)+1
        self.init_plot = self.storage_map.reshape(self.storage_shape) 
        return np.array(self.storage_map).copy()

    def vec_reset(self, num_envs=None):
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._storage_maps = np.vstack(list(map(np.random.permutation,[self.num_products]*num_envs)))+1
        self.storage_map = self._storage_maps[0]
        self.init_plot = self.storage_map.reshape(self.storage_shape) 
        return np.array(self._storage_maps).copy()

    def get_bin_coordinate(self,bin_id):
        '''
        Given a bin number, this gives the location of the bin. This can be useful to to calculate distance between bin to exit.
        '''
        if len(self.storage_shape) == 3:
            a,b,c = self.storage_shape
            x, y, z = bin_id//(b*c),  bin_id%(b*c)//c, bin_id%c
            return x, y, z
        elif len(self.storage_shape) == 2:
            a,b = self.storage_shape
            x, y = bin_id//b,  bin_id%b
            return x, y
        elif len(self.storage_shape) == 1:
            return bin_id
    
    def get_distance_to_exit(self,bin_id = None):
        if bin_id is None:
            return np.vstack(self.get_bin_coordinate(np.arange(self.num_products))).sum(axis=0) + self.dist_origin_to_exit
        else:
            return sum(self.get_bin_coordinate(bin_id)) + self.dist_origin_to_exit

    def get_orders(self, num_envs=1):
        if num_envs == 1:
            order = np.random.binomial(1, self.dist_param)
        else:
            order = np.random.binomial(1, np.repeat(self.dist_param, num_envs).reshape((self.num_products, num_envs))).T
        return order

    def step(self, action=None):
        '''
        Action should be a tuple (x, y), which indicates that good in bin number x and bin number y should switch.
        '''
        assert action is None or (action[0] < action[1] and action[1] < self.num_products and action[0] > -1), f"Invalid action {action}!"

        storage_map = self.storage_map
        exchange_cost = 0
        order = self.get_orders()
        if (action is not None):
            storage_map[action[0]], storage_map[action[1]] = storage_map[action[1]], storage_map[action[0]]
            exchange_cost +=np.abs(np.array(self.get_bin_coordinate(action[0]))-np.array(self.get_bin_coordinate(action[1]))).sum()
        self.storage_map = storage_map
        return self.storage_map.copy(), order, exchange_cost


    def vec_step(self, actions):
        # actions is a list of length n either 2-tuple or None
        assert np.array(list(map((lambda action: action is None or (action[0] < action[1] and action[1] < self.num_products and action[0] > -1)), actions))).all()
        assert self._storage_maps is not None
        range_n = np.arange(self._num_envs)
        actions = np.array([action if action is not None else (0, 0) for action in actions])
        self._storage_maps[range_n, actions[:,0]], self._storage_maps[range_n, actions[:,1]] =\
             self._storage_maps[range_n, actions[:,1]], self._storage_maps[range_n, actions[:,0]] 
        orders = self.get_orders(num_envs=self._num_envs)
        exchange_costs = np.abs(np.array(self.get_bin_coordinate(actions[:,0]))-np.array(self.get_bin_coordinate(actions[:,1]))).sum(axis=0)
        self.storage_map = self._storage_maps[0]
        return np.array(self._storage_maps).copy(),orders, exchange_costs

    def set_storage_map(self, storage_map):
        self.storage_map = storage_map.copy()

    def vec_set_storage_map(self, storage_maps):
        self._num_envs = len(storage_maps)
        self._storage_maps = storage_maps.copy()

    def render(self, mode='human', iteration=None):
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            self._ax.tick_params(
                axis='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)  # labels along the bottom edge are off
            self._ax.set_aspect('equal')
            self._canvas = FigureCanvas(self._fig)
        current_map = self.storage_map.reshape(self.storage_shape)
        data = self.cmap(self.dist_param[current_map-1]) 
        data = self.upsample(data,self._scale) 
        for ix,iy in np.ndindex(self.init_plot.shape):
                number = current_map[ix,iy]
                self.add_numbers_on_plot(number,data[ix*16:(ix+1)*16,iy*16:(iy+1)*16,:3])
        self._render = self._ax.imshow(data,animated=True)
        # self._render.set_data(data)
        if iteration is not None:
            self._ax.set_title('Iteration %d' % iteration)
        self._canvas.draw()
        self._canvas.flush_events()
        time.sleep(self.dt)

        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image

    def upsample(self, image, scale):
        up_image = np.repeat(image, scale, axis=0)
        up_image = np.repeat(up_image, scale, axis=1)
        return up_image

    def add_numbers_on_plot(self, number, box):
        number = str(number)
        imageRGB = Image.new('RGB', (self._scale, self._scale))
        draw  = ImageDraw.Draw(imageRGB)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf',size = 12)
        w, h  = draw.textsize(number, font=font)
        draw.text(((self._scale - w)/2, (self._scale - h)/2), number)
        p = 1-np.array(imageRGB)/255
        box[np.where(p==0)]= p[np.where(p==0)]

    def close(self):
        plt.close()
        self._fig = None

if __name__ == "__main__":
    a = ASRSEnv((2,3))
    a.render()
    a.step((2,4))

import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import time
from PIL import ImageDraw,Image,ImageFont
import logger


class ASRSEnv(object):

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

    def __init__(self, storage_shape, order_process, origin_coord=None, seed=42):
        self.order_process = order_process
        self.map_random = np.random.RandomState()
        self.set_seed(seed)
        assert len(storage_shape) <= 3, "storage_shape length should be <= 3"
        self.storage_shape = storage_shape
        self.obs_dim = 1
        self.num_products = np.array(storage_shape).prod()
        self.num_maps = math.factorial(self.num_products)
        self.num_actions = int(self.num_products * (self.num_products - 1) / 2 + 1)
        assert (origin_coord is None) or (len(storage_shape) == len(origin_coord)), "origin_coord does not have correct dimensions"
        if origin_coord:
            if np.array(origin_coord).ndim == 2:
                self.origin_coords = np.array(origin_coord)
            else:
                self.origin_coords = np.array(origin_coord)[np.newaxis]
        else:
            self.origin_coords = np.zeros((len(storage_shape),storage_shape[0])).astype(int)
            self.origin_coords[0] = np.arange(storage_shape[0])
            self.origin_coords = self.origin_coords.T
            # np.zeros(len(storage_shape)).astype(int)
            # Default is (:, 0, 0)
        self.dist_origin_to_exit = 1 # Distance from origin to exit

        assert order_process.num_products == self.num_products, "Number of products in order process need to match number of storage bins"

        self.dynamic_order = order_process.dynamic_order
        self.reset()

        self._fig = None
        # self.cmap = matplotlib.cm.get_cmap('Spectral')
        self.cmap = matplotlib.cm.get_cmap('coolwarm')
        self.dt = .2
        self._scale = 16
        self.vectorized = True
        self.__name__ = 'ASRSEnv'

    def set_seed(self, seed=None):
        if seed:
            self.seed_num = seed
            self.order_process.set_seed(self.seed_num+1000)
        self.map_random.seed(self.seed_num+100)

    def reset(self):
        self.set_seed()
        self._storage_maps = None
        self._num_envs = None
        self.order_process.reset()
        self.storage_map = self.map_random.permutation(self.num_products)+1
        return np.array(self.storage_map).copy()

    def vec_reset(self, num_envs=None):
        self.set_seed()
        self.order_process.reset()
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._storage_maps = np.vstack(list(map(self.map_random.permutation,[self.num_products]*num_envs)))+1
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
            coords = self.get_bin_coordinate(np.arange(self.num_products))
        else:
            coords = self.get_bin_coordinate(bin_id)
        
        dist_to_each_exit = []
        for origin in self.origin_coords:
            dist_to_each_exit.append(self.get_distance_between_coord(coords,np.vstack(origin)) + self.dist_origin_to_exit)
        return np.array(dist_to_each_exit).min(axis=0)

    def get_distance_between_coord(self, coord1, coord2):
        return np.abs(np.array(coord1)-np.array(coord2)).sum(axis=0)

    def get_order_sequence(self, num_period=1):
        # Can only generate order sequences for 1 environments
        order_sequence = np.zeros((num_period, self.num_products))
        p_sequence = np.zeros((num_period, self.num_products))
        for t in range(num_period):
            order = self.order_process.get_orders(num_envs=1)
            order_sequence[t] = order
            p_sequence[t] = self.order_process.dist_param
        return order_sequence, p_sequence

    def step(self, action=None, rollout = True):
        '''
        Action should be a tuple (x, y), which indicates that good in bin number x and bin number y should switch.
        '''
        assert action is None or (action[0] < action[1] and action[1] < self.num_products and action[0] > -1), f"Invalid action {action}!"

        storage_map = self.storage_map
        exchange_cost = 0
        if rollout:
            order = self.order_process.get_orders()
        else:
            order = None
        if (action is not None):
            storage_map[action[0]], storage_map[action[1]] = storage_map[action[1]], storage_map[action[0]]
            exchange_cost += self.get_distance_between_coord(self.get_bin_coordinate(action[0]), self.get_bin_coordinate(action[1]))
        self.storage_map = storage_map
        return self.storage_map.copy(), order, exchange_cost


    def vec_step(self, actions, rollout = True):
        # actions is a list of length n either 2-tuple or None
        assert np.array(list(map((lambda action: action is None or (action[0] < action[1] and action[1] < self.num_products and action[0] > -1)), actions))).all()
        assert self._storage_maps is not None
        actions = np.array([action if action is not None else (0, 0) for action in actions])
        self._storage_maps = self.vec_next_storage(self._storage_maps, actions)
        if rollout:
            orders = self.order_process.get_orders(num_envs=self._num_envs)
        else:
            orders = None
        exchange_costs = self.get_distance_between_coord(self.get_bin_coordinate(actions[:,0]), self.get_bin_coordinate(actions[:,1]))
        return self._storage_maps.copy(),orders, exchange_costs
    
    def vec_next_storage(self, storage_maps, actions):
        next_storage_maps = storage_maps.copy()
        range_n = np.arange(next_storage_maps.shape[0])
        next_storage_maps[range_n, actions[:,0]], next_storage_maps[range_n, actions[:,1]] =\
             next_storage_maps[range_n, actions[:,1]], next_storage_maps[range_n, actions[:,0]] 
        return next_storage_maps

    def set_state(self, storage_map):
        self.storage_map = storage_map.copy()

    def vec_set_state(self, storage_maps):
        self._num_envs = len(storage_maps)
        self._storage_maps = storage_maps.copy()

    def render(self, mode='human', iteration=None):
        assert len(self.storage_shape) == 2, "Storage map need to be 2-d in order to render"
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
        if self._storage_maps is not None:
            current_map = self._storage_maps[0].reshape(self.storage_shape)
        else:
            current_map = self.storage_map.reshape(self.storage_shape)
        # if self.dynamic_order:
        #     data = self.cmap((self.long_term_2p[self.season]/2)[current_map-1])
        # else:
        #     data = self.cmap(self.order_process.dist_param[current_map-1]) 
        data = self.cmap(self.order_process.dist_param[current_map-1]) 
        data = self.upsample(data,self._scale) 
        for ix,iy in np.ndindex(self.storage_shape):
                number = current_map[ix,iy]
                box = data[ix*self._scale:(ix+1)*self._scale,iy*self._scale:(iy+1)*self._scale,:3]
                self.add_numbers_on_plot(number,box)
        for origin_num in range(self.origin_coords.shape[0]):        
            self.mark_exit_on_plot(data, origin_num)
        self._render = self._ax.imshow(data,animated=True)
        # self._render.set_data(data)
        if iteration is not None:
            self._ax.set_title('Iteration %d, time %d' % (iteration, self.order_process.age))
        self._canvas.draw()
        self._canvas.flush_events()
        time.sleep(self.dt)

        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image
        if mode == 'human':
            s, (width, height) = self._canvas.print_to_buffer()
            plt.imshow(np.fromstring(s,dtype='uint8').reshape(int(height), int(width), 4))     

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

    def mark_exit_on_plot(self, plot, origin_num):
        box = plot[self.origin_coords[origin_num][0]*self._scale:(self.origin_coords[origin_num][0]+1)*self._scale,self.origin_coords[origin_num][1]*self._scale:(self.origin_coords[origin_num][1]+1)*self._scale,:3]
        border = np.ones((self._scale,self._scale,3),dtype=bool)
        border[1:-1,1:-1,:] = False
        color = np.array([255, 215, 0])/255
        rgb_patch = np.ones((self._scale,self._scale, 3), dtype=np.uint8)
        rgb_patch= rgb_patch*color
        box[border]= rgb_patch[border]

    def close(self):
        plt.close()
        self._fig = None


if __name__ == "__main__":
    a = ASRSEnv((2,3))
    a.get_distance_to_exit()
    a.render()
    a.step((2,4))


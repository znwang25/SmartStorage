import numpy as np
from envs import ASRSEnv
from collections.abc import Iterable
import scipy.sparse as sparse
import itertools
from utils.utils import SparseArray
import logger

class MapAsPicEnv(object):
    """
    Description:
    This wrapper constructs state and provide various utilities for reinforcement learner to use to interact with the underlying environment ASRSEnv. 

    State:
        Current Storage Map    np.array()   (M,)     
        Outstanding order   np.array()   (M,)     
    Actions:
        Num	of Action: M Choose 2 + 1
        e.g  (a , b) switch good in bin a with good bin b 
             or None do nothing

    Cost:
        Cost is delay_cost + self.exchange_cost_weight*exchange_cost

    Starting State:
        A random permutation.
    """

    def __init__(self,
                 env,alpha=10,discount = 0.99,scale = 16):
        logger.info("Initiating MapAsPicEnv")
        self._wrapped_env = env
        self.scale = scale
        self.num_products = env.num_products
        self.storage_shape = env.storage_shape
        self.state_shape = np.array(self.storage_shape).astype(int) * scale
        self.obs_dim = env.obs_dim
        self.exchange_cost_weight = alpha
        self.discount = discount
        self.distance = env.get_distance_to_exit()
        self.max_distance = self.distance.max()

        # self.completed_order_mask = self.get_completed_order_mask()        
        all_permutations = list(itertools.permutations(range(1,self.num_products+1)))
        self.num_maps = env.num_maps
        self.num_states = self.num_maps # To be changed
        self.id_to_map_dict = dict(zip(range(self.num_maps),all_permutations))
        self.map_to_id_dict = dict(zip(all_permutations,range(self.num_maps)))
        all_actions = list(itertools.combinations(range(self.num_products),2))+[None]
        self.num_actions = env.num_actions
        self.id_to_action_dict = dict(zip(range(self.num_actions),all_actions))
        self.vectorized = True

    def step(self, action):
        next_storage_map, order, exchange_cost = self._wrapped_env.step(action)
        delay_cost = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*order[next_storage_map-1]).sum()
        cost = delay_cost + self.exchange_cost_weight*exchange_cost
        next_state = self.storage_map_to_state(next_storage_map)
        reward = - cost
        return next_state, reward, delay_cost

    def reset(self, init=False):
        next_storage_map = self._wrapped_env.reset()
        next_state = np.squeeze(self.storage_map_to_state(next_storage_map),axis=0)
        return next_state

    def vec_reset(self, num_envs):
        next_storage_maps = self._wrapped_env.vec_reset(num_envs)
        next_states = self.storage_map_to_state(next_storage_maps)
        return next_states

    def vec_step(self, actions):
        p = self._wrapped_env.dist_param
        num_acts = actions.shape[0]
        next_storage_maps, orders, exchange_costs = self._wrapped_env.vec_step(actions)
        delay_costs = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*p[next_storage_maps-1]).sum(axis=1)
        # delay_costs = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*orders[np.repeat(np.arange(num_acts),orders.shape[1]).reshape(num_acts,-1),next_storage_maps[np.arange(num_acts)]-1]).sum(axis=1)
        costs = delay_costs + self.exchange_cost_weight*exchange_costs
        next_states = self.storage_map_to_state(next_storage_maps)
        rewards = - costs
        return next_states, rewards, delay_costs

    def set_state(self, state):
        storage_map = self.downsample(state,self.scale).reshape(-1)
        self._wrapped_env.set_state(storage_map)

    def vec_set_state(self, states):
        storage_maps = self.downsample(states,self.scale).reshape(states.shape[0],-1)
        self._wrapped_env.vec_set_state(storage_maps)


    def storage_map_to_state(self,storage_maps):
        storage_maps = storage_maps.reshape(-1, *self.storage_shape)
        states= self.upsample(storage_maps,self.scale)
        return states

    def sample_states(self, batch_size):
        id_m_s = np.random.randint(0, self.num_states, size=(batch_size,))
        storage_maps = np.array(list(map(self.get_map_from_id, id_m_s)))
        return self.storage_map_to_state(storage_maps)

    def sample_actions(self, num_acts, all = False):
        if all:
            id_a_s = np.arange(self.num_actions)
        else:
            id_a_s = np.random.randint(0, self.num_actions, size=(num_acts,))
        actions = np.array(list(map(self.get_action_from_id, id_a_s)))
        return actions

    def get_map_from_id(self, id_m):
        """
        Get map configuration from id
        :param id_m:
        :return: map_storage
        """
        return self.id_to_map_dict[id_m]

    def get_id_from_map(self, map_storage):
        """
        Get id of the map configuration
        :param map_storage: tuple or np.array 
        :return: id_m
        """
        return self.map_to_id_dict[tuple(map_storage)]

    def get_action_from_id(self, id_a):
        """
        Get action from id
        :param id_a:
        :return:
        """
        return self.id_to_action_dict[id_a]

    def upsample(self, image, scale):
        up_image = np.repeat(image, scale, axis=-2)
        up_image = np.repeat(up_image, scale, axis=-1)
        return up_image

    def downsample(self, image, scale):
        down_image = image.take(np.arange(image.shape[-2], step=scale),axis=-2).take(np.arange(image.shape[-1], step=scale),axis=-1)
        return down_image

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        # orig_attr = self._wrapped_env.__getattribute__(attr)
        if hasattr(self._wrapped_env, '_wrapped_env'):
            orig_attr = self._wrapped_env.__getattr__(attr)
        else:
            orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


if __name__ == '__main__':
    a = ASRSEnv((3,3,4))
    a = ASRSEnv((2,1,1))
    b=MapAsPicEnv(a)
    print(b.step())
    for i in range(10):
        print(b.step())
        if i%2==0:
            print(b.step((0,5)))

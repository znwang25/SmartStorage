import numpy as np
from envs import ASRSEnv
from collections.abc import Iterable
import scipy.sparse as sparse
import itertools
from utils.utils import SparseArray
import logger

class ExpandStateWrapper(object):
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
                 env,alpha=10,discount = 0.99):
        logger.info("Initiating ExpandStateWrapper")
        self._wrapped_env = env
        self.num_products = env.num_products
        self.storage_shape = env.storage_shape
        self.obs_dim = env.obs_dim
        self.exchange_cost_weight = alpha
        self.discount = discount
        self.distance = env.get_distance_to_exit()
        self.max_distance = self.distance.max()

        self.completed_order_mask = self.get_completed_order_mask()        
        all_permutations = list(itertools.permutations(range(1,self.num_products+1)))
        self.num_maps = len(all_permutations)
        self.num_states = self.num_maps # To be changed
        self.id_to_map_dict = dict(zip(range(self.num_maps),all_permutations))
        self.map_to_id_dict = dict(zip(all_permutations,range(self.num_maps)))
        all_actions = list(itertools.combinations(range(self.num_products),2))+[None]
        self.num_actions = env.num_actions
        self.id_to_action_dict = dict(zip(range(self.num_actions),all_actions))
        logger.info('Starting transitions and rewards')
        self.get_transitions_rewards()
        logger.info("Env finished")
        self.reset(init=True)

        self.vectorized = True

    def get_transitions_rewards(self):
        self.transitions = SparseArray(self.num_states, self.num_actions, 'nn', self.num_states)
        self.rewards = SparseArray(self.num_states, self.num_actions, 'nn', self.num_states)
        _values_transition = np.ones((self.num_states, self.num_actions))
        _values_cost = np.zeros((self.num_states, self.num_actions))
        _idxs = np.zeros((self.num_states, self.num_actions)).astype(int)
        p = self._wrapped_env.dist_param
        for id_m in range(self.num_states):
            next_storage_maps, next_storage_ids, exchange_costs = self.vec_next_map_id_excost(id_m, np.arange(self.num_actions))
            _idxs[id_m,:] = next_storage_ids
            _values_cost[id_m,:] += self.exchange_cost_weight*exchange_costs
            _values_cost[id_m,:] += ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*p[next_storage_maps-1]).sum(axis=1)
        self.transitions._values, self.transitions._idxs = np.expand_dims(_values_transition, axis=2), np.expand_dims(_idxs, axis=2)
        self.rewards._values, self.rewards._idxs = -np.expand_dims(_values_cost, axis=2), np.expand_dims(_idxs, axis=2)

    def next_map_id_excost(self, id_m, id_a):
        storage_map = list(self.get_map_from_id(id_m))
        action =self.get_action_from_id(id_a)
        exchange_cost = 0
        if (action is not None) and (action[1]!=action[0]):
            storage_map[action[0]], storage_map[action[1]] = storage_map[action[1]], storage_map[action[0]]
            exchange_cost = self._wrapped_env.get_distance_between_coord(self._wrapped_env.get_bin_coordinate(action[0]),self._wrapped_env.get_bin_coordinate(action[1]))
        return self.get_id_from_map(storage_map),exchange_cost
    
    def vec_next_map_id_excost(self, id_m, id_a_s):
        storage_maps = np.array(list(self.get_map_from_id(id_m))) * np.ones((len(id_a_s),1)).astype(int)
        actions = np.array(list(map(self.get_action_from_id, id_a_s)))
        actions = np.array([action if action is not None else (0, 0) for action in actions])
        next_storage_maps = self._wrapped_env.vec_next_storage(storage_maps, actions)
        next_storage_ids = np.array(list(map(self.get_id_from_map, next_storage_maps)))
        exchange_costs = self._wrapped_env.get_distance_between_coord(self.get_bin_coordinate(actions[:,0]),self.get_bin_coordinate(actions[:,1]))
        return next_storage_maps, next_storage_ids, exchange_costs

    def get_completed_order_mask(self):
        mask = (self.max_distance-np.arange(self.max_distance).reshape((-1,1)) == self.distance).astype(int)
        return sparse.csr_matrix(mask)

    def step(self, id_a):
        action = self.get_action_from_id(id_a)
        next_storage_map, order, exchange_cost = self._wrapped_env.step(action)
        delay_cost = self.outstanding_order.sum()
        cost = delay_cost + self.exchange_cost_weight*exchange_cost
        self.outstanding_order += order
        completed_order = self.map_history.multiply(self.completed_order_mask)
        if completed_order.count_nonzero() != 0:
            completed_order = np.array(completed_order[completed_order.nonzero()]).ravel().astype(int)-1
            self.outstanding_order -= np.bincount(completed_order, minlength=self.num_products)
        self.map_history = sparse.vstack([self.map_history,sparse.csr_matrix(next_storage_map * order[next_storage_map-1])])[1:]
        reward = -cost
        return self.get_id_from_map(next_storage_map), reward, delay_cost

    def reset(self, init=False):
        if not init:
            self._wrapped_env.reset()
        self.map_history = sparse.csr_matrix(np.zeros((self.max_distance,self.num_products))) 
        self.outstanding_order = np.zeros(self.num_products).astype(int)
        next_storage_id, _, _= self.step(self.num_actions-1) # To get first order
        self._outstanding_orders = None
        self._map_histories = None
        return next_storage_id

    def vec_reset(self, num_envs):
        self._wrapped_env.vec_reset(num_envs)
        self._outstanding_orders = np.zeros((num_envs,self.num_products)).astype(int)
        self._map_histories = sparse.csr_matrix(np.zeros((self.max_distance*num_envs,self.num_products))) 
        next_storage_ids, _, _= self.vec_step([self.num_actions-1]*num_envs)
        return next_storage_ids

    def vec_step(self, id_a_s):
        assert self._outstanding_orders is not None
        actions = np.array(list(map(self.get_action_from_id, id_a_s)))
        next_storage_maps, orders, exchange_costs = self._wrapped_env.vec_step(actions)
        next_storage_ids = np.array(list(map(self.get_id_from_map, next_storage_maps)))
        delay_cost = self._outstanding_orders.sum(axis = 1)
        costs = delay_cost + self.exchange_cost_weight*exchange_costs
        self._outstanding_orders += orders
        num_envs = self._wrapped_env._num_envs
        completed_orders = self._map_histories.multiply(np.repeat(self.completed_order_mask.toarray(),num_envs,axis=0)).tocsr()
        
        # Vectorized bincount 
        if completed_orders.count_nonzero() != 0:
            nonzero_idx, _ = completed_orders.nonzero()
            temp = completed_orders[completed_orders.nonzero()]+(self.num_products)*(nonzero_idx%num_envs)
            temp = np.array(temp).ravel().astype(int)-1
            completed_orders_count = np.bincount(temp,minlength=self.num_products*num_envs).reshape(-1,self.num_products)
            self._outstanding_orders -= completed_orders_count
        orders_sortby_bin = orders[np.repeat(np.arange(num_envs),self.num_products).reshape(num_envs,self.num_products),next_storage_maps-1]
        self._map_histories = sparse.vstack([self._map_histories,sparse.csr_matrix(next_storage_maps * orders_sortby_bin)])[num_envs:]
        rewards = -costs
        return next_storage_ids, rewards, delay_cost

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
    b=ExpandStateWrapper(a)
    print(b.step())
    for i in range(10):
        print(b.step())
        if i%2==0:
            print(b.step((0,5)))

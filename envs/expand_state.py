import numpy as np
from envs import ASRSEnv
from collections.abc import Iterable
import scipy.sparse as sparse
import itertools
from utils.utils import SparseArray

class ExpandStateWrapper(object):
    """
    Description:
    This wrapper construct state and provide various utilities for re-enforcement learner to use to interact with the underlying environment. 

    State:
        Current Storage Map    np.array()   (M,)     
        Outstanding order   np.array()   (M,)

    Actions:
        Num	of Action: M Choose 2 + 1
        e.g  (a , b) switch bin a with bin b 
             or do nothing

    Cost:
        Cost is delay_cost + self.exchange_cost_weight*exchange_cost

    Starting State:
        A random permutation.
    """

    def __init__(self,
                 env,alpha=10,discount = 0.99):
        print("Initiating Env")
        self._wrapped_env = env
        self._costs = None
        self.num_products = env.num_products
        self.max_distance = env.max_distance
        self.exchange_cost_weight = alpha
        self.discount = discount
        self.distance = env.get_distance_to_exit()
        self.map_history = sparse.csr_matrix(np.zeros((self.max_distance,self.num_products))) 
        self.outstanding_order = np.zeros(self.num_products).astype(int)
        self.completed_order_mask = self.get_compelted_order_mask()
        all_permutations = list(itertools.permutations(range(1,self.num_products+1)))
        self.num_maps = len(all_permutations)
        self.num_states = self.num_maps # To be changed
        self.obs_dim = 1
        self.id_to_map_dict = dict(zip(range(self.num_maps),all_permutations))
        self.map_to_id_dict = dict(zip(all_permutations,range(self.num_maps)))
        all_actions = list(itertools.combinations(range(self.num_products),2))+[None]
        self.num_actions = len(all_actions)
        self.action_dim = 1
        self.id_to_action_dict = dict(zip(range(self.num_actions),all_actions))
        print('Starting transitions')
        self.get_transitions()
        print('Starting rewards')
        self.get_rewards()
        self.step(self.num_actions-1) # To get first order
        # self._states = self._wrapped_env._storage_maps
        print("Env finished")

    def get_transitions(self):
        self.transitions = SparseArray(self.num_states, self.num_actions, 'nn', self.num_states)        
        for id_m in range(self.num_states):
            for id_a in range(self.num_actions):
                id_m_next,_ = self.next_map_id(id_m,id_a)
                self.transitions[id_m,id_a,id_m_next] = 1

        self.transitions = SparseArray(self.num_states, self.num_actions, 'nn', self.num_states)        
        for id_m in range(self.num_states):
            for id_a in range(self.num_actions):
                id_m_next,_ = self.next_map_id(id_m,id_a)
                self.transitions[id_m,id_a,id_m_next] = 1

    def get_rewards(self):
        self.rewards = SparseArray(self.num_states, self.num_actions, 'nn', self.num_states)        
        for id_m in range(self.num_states):
            for id_a in range(self.num_actions):
                id_m_next, exchange_cost = self.next_map_id(id_m,id_a)
                next_map = np.array(self.get_map_from_id(id_m_next))
                p = self._wrapped_env.dist_param
                expected_1_period_delay_cost = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*p[next_map-1]).sum()
                self.rewards[id_m,id_a,id_m_next] = - (expected_1_period_delay_cost + self.exchange_cost_weight*exchange_cost)

    def next_map_id(self, id_m, id_a):
        storage_map = list(self.get_map_from_id(id_m))
        action =self.get_action_from_id(id_a)
        exchange_cost = 0
        if (action is not None) and (action[1]!=action[0]):
            storage_map[action[0]], storage_map[action[1]] = storage_map[action[1]], storage_map[action[0]]
            exchange_cost = np.abs(np.array(self._wrapped_env.get_bin_coordinate(action[0]))-np.array(self._wrapped_env.get_bin_coordinate(action[1]))).sum()
        return self.get_id_from_map(tuple(storage_map)),exchange_cost

    def get_compelted_order_mask(self):
        mask = np.zeros((self.max_distance,self.num_products))
        for i in range(self.max_distance): 
            mask[i] = (self.max_distance-i == self.distance)
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
        return self.get_id_from_map(tuple(next_storage_map)), self.outstanding_order.copy(), cost

    def reset(self):
        storage_map = self._wrapped_env.reset()
        return self.get_id_from_map(tuple(storage_map))
        

    # def vec_step(self, actions):
    #     next_storage_maps, orders, costs = self._wrapped_env.vec_step(actions)
    #     pass

    # def vec_set_state(self, storage_maps):
    #     self._wrapped_env.vec_set_storage_maps(storage_maps)
    #     pass

    # def vec_reset(self, num_envs):
    #     storage_maps = self._wrapped_env.vec_reset(num_envs)
    #     pass

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
        :param map_storage:
        :return: id_m
        """
        return self.map_to_id_dict[map_storage]

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

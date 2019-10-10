import numpy as np
from envs.asrs_env import ASRSEnv
from collections.abc import Iterable
import scipy.sparse as sparse
import itertools

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
                 env,alpha=10):
        self._wrapped_env = env
        self._costs = None
        self.num_products = env.num_products
        self.max_distance = env.max_distance
        self.distance = env.get_distance_to_exit()
        self.exchange_cost_weight = alpha
        self.map_history = sparse.csr_matrix(np.zeros((self.max_distance,self.num_products))) 
        self.outstanding_order = np.zeros(self.num_products).astype(int)
        self.completed_order_mask = self.get_compelted_order_mask()
        self.step()
        all_permutations = list(itertools.permutations(range(self.num_products)))
        self.id_to_map_dict = dict(zip(range(len(all_permutations)),all_permutations))
        self.map_to_id_dict = dict(zip(all_permutations,range(len(all_permutations))))
        all_actions = list(itertools.permutations(range(self.num_products),2))+[None]
        self.id_to_action_dict = dict(zip(range(len(all_actions)),all_actions))

        # self._states = self._wrapped_env._storage_maps

    def get_compelted_order_mask(self):
        mask = np.zeros((self.max_distance,self.num_products))
        for i in range(self.max_distance): 
            mask[i] = (self.max_distance-i == self.distance)
        return sparse.csr_matrix(mask)

    def step(self, action=None):
        next_storage_map, order, exchange_cost = self._wrapped_env.step(action)
        delay_cost = self.outstanding_order.sum()
        cost = delay_cost + self.exchange_cost_weight*exchange_cost
        self.outstanding_order += order
        completed_order = self.map_history.multiply(self.completed_order_mask)
        if completed_order.count_nonzero() != 0:
            completed_order = np.array(completed_order[completed_order.nonzero()]).ravel().astype(int)-1
            self.outstanding_order -= np.bincount(completed_order, minlength=self.num_products)
        self.map_history = sparse.vstack([self.map_history,sparse.csr_matrix(next_storage_map * order[next_storage_map-1])])[1:]
        return next_storage_map, self.outstanding_order.copy(), cost

    def reset(self):
        storage_map = self._wrapped_env.reset()
        pass

    def vec_step(self, actions):
        next_storage_maps, orders, costs = self._wrapped_env.vec_step(actions)
        pass

    def vec_set_state(self, storage_maps):
        self._wrapped_env.vec_set_storage_maps(storage_maps)
        pass

    def vec_reset(self, num_envs):
        storage_maps = self._wrapped_env.vec_reset(num_envs)
        pass

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
        return self.map_id_dict[map_storage]

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
    b=ExpandStateWrapper(a)
    print(b.step())
    for i in range(10):
        print(b.step())
        if i%2==0:
            print(b.step((0,5)))

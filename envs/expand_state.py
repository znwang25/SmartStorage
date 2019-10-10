import numpy as np
from envs.asrs_env import ASRSEnv
from collections.abc import Iterable
import scipy.sparse as sparse

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

    def get_state_from_id(self, id_s):
        """
        Get continuous state from id
        :param id_s:
        :return:
        """
        if self._disc_state:
            return id_s
        else:
            vec = self.get_coordinates_from_id(id_s)
            return self.state_points[range(self.obs_dims), vec]

    def get_action_from_id(self, id_a):
        """
        Get continous action from id
        :param id_a:
        :return:
        """
        if self._disc_act:
            return id_a
        else:
            vec = self.get_coordinates_from_id(id_a, state=False)
            return self.act_points[range(self.act_dims), vec]

    def get_coordinates_from_id(self, idx, state=True, base=None):
        """
        Get position in the grid from id
        :param idx:
        :param state:
        :param base:
        :return:
        """
        size = self.obs_dims if state else self.act_dims
        if isinstance(idx, Iterable): # probably if it's iterable
            vec = np.zeros((len(idx), size))
        else:
            vec = np.zeros((size,), dtype=np.int)

        num, i,  = idx, 0
        if base is None:
            base = self._state_bins_per_dim if state else self._act_bins_per_dim
        else:
            assert type(base) == int
            base = np.ones((size,), dtype=np.int) * base
        for i in range(size):
            vec[..., i] = num % base[i]
            num = num//base[i]
            i += 1
        return vec.astype(np.int)

    def get_id_from_coordinates(self, vec, state=True):
        """
        Get id from position in the grid
        :param vec:
        :param state:
        :return:
        """
        base_transf = self._state_base_transf if state else self._act_base_transf
        return np.squeeze(np.sum(vec * base_transf, axis=-1).astype(int))

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

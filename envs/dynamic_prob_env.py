import numpy as np
from collections.abc import Iterable
import scipy.sparse as sparse
import itertools
from utils.utils import random_choice_noreplace, random_permutation
import logger


class DynamicProbEnv(object):
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
                 env,
                 demand_predictor,
                 num_p_in_states = 10, 
                 alpha=1, discount=0.99):
        logger.info("Initiating DynamicProbEnv")
        self._wrapped_env = env
        self.demand_predictor = demand_predictor
        self.num_products = env.num_products
        self.storage_shape = env.storage_shape
        # self.state_shape = (
        #     int(self.num_products*(2+num_p_in_states) + self.num_products*(self.num_products-1)/2),)
        # self.state_shape = (
        #     int(self.num_products*(2+num_p_in_states)),)
        self.state_shape = (
            int(self.num_products*(1+num_p_in_states)),)

        self.obs_dim = 1
        self.exchange_cost_weight = alpha
        self.discount = discount
        self.distance = env.get_distance_to_exit()
        self.max_distance = self.distance.max()
        self.num_maps = env.num_maps
        self.num_states = self.num_maps  # To be changed
        self.num_p_in_states = num_p_in_states

        self.num_actions = env.num_actions
        self.vectorized = True
        self.rnn_lookback = demand_predictor.look_back
        logger.info("Getting order_history")
        self.order_history, _ = self._wrapped_env.get_order_sequence(num_period=demand_predictor.look_back+self.num_p_in_states)
        self.p_state = self.demand_predictor.get_predicted_p(self.order_history, preprocess = True)
        self.p_features_set = self.demand_predictor.sliding_window(self.demand_predictor.buffer_p_sequence_hat,self.num_p_in_states+1)
        self.p_state_init = self.p_state # The p value to reset to.
        logger.info("Finished env set up")

    def update_order_history(self, order):
        self.order_history = np.vstack([self.order_history, order])[1:]
        self.p_state = np.vstack([self.p_state, self.demand_predictor.get_predicted_p(self.order_history[-self.rnn_lookback:])])[1:]

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]
        next_storage_map, order, exchange_cost = self._wrapped_env.step(action)
        p_next = self.p_state.reshape(-1)
        # Use true p value to calculate rewards/costs
        if self._wrapped_env.dynamic_order:
            p = self._wrapped_env.order_process.long_term_2p[self._wrapped_env.order_process.season]/2 # next period p
        else:
            p = self._wrapped_env.order_process.dist_param # next period p

        delay_cost = ((self.discount/(1-self.discount)*(1-self.discount **
                                                        self.distance))*p[next_storage_map-1]).sum()
        cost = delay_cost + self.exchange_cost_weight*exchange_cost
        next_state = self.storage_map_to_state(next_storage_map, p_next)
        reward = - cost
        return next_state, reward, delay_cost

    def reset(self, init=False):
        next_storage_map = self._wrapped_env.reset()
        next_state = self.storage_map_to_state(next_storage_map,self.p_state_init.reshape(-1))
        return next_state

    def vec_reset(self, num_envs):
        next_storage_maps = self._wrapped_env.vec_reset(num_envs)
        next_states = self.storage_map_to_state(next_storage_maps, np.repeat(self.p_state_init.reshape(1,-1),num_envs,axis=0))
        return next_states

    def vec_rollout_step(self, actions):
        num_acts = actions.shape[0]
        next_storage_maps, orders, exchange_costs = self._wrapped_env.vec_step(
            actions)
        self.update_order_history(orders[0])
        p_next = np.repeat(self.p_state.reshape(1,-1),num_acts,axis=0)

        # Use true p value to calculate rewards/costs
        if self._wrapped_env.dynamic_order:
            p_hat = self._wrapped_env.order_process.long_term_2p[self._wrapped_env.order_process.season][np.newaxis,:]/2 # next period p_hat
        else:
            p_hat = self._wrapped_env.order_process.dist_param[np.newaxis,:] # next period p_hat
        delay_costs_hat = ((self.discount/(1-self.discount)*(1-self.discount **
                                                             self.distance))*np.take_along_axis(p_hat,next_storage_maps-1,axis=-1)).sum(axis=1)
        costs_hat = delay_costs_hat + self.exchange_cost_weight*exchange_costs
        rewards_hat = -costs_hat

        next_states = self.storage_map_to_state(next_storage_maps, p_next)
        return next_states, rewards_hat, delay_costs_hat

    def vec_step(self, actions, p_next, rollout = False):
        num_acts = actions.shape[0]
        next_storage_maps, orders, exchange_costs = self._wrapped_env.vec_step(
            actions, rollout = rollout)
        p_hat = p_next.reshape(-1,self.num_p_in_states, self.num_products)[:,-1,:]
        delay_costs_hat = ((self.discount/(1-self.discount)*(1-self.discount **
                                                             self.distance))*np.take_along_axis(p_hat,next_storage_maps-1,axis=-1)).sum(axis=1)
        costs_hat = delay_costs_hat + self.exchange_cost_weight*exchange_costs
        rewards_hat = -costs_hat
        next_states = self.storage_map_to_state(next_storage_maps, p_next)
        return next_states, rewards_hat, delay_costs_hat

        # delay_costs = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*p[next_storage_maps-1]).sum(axis=1)
        # # delay_costs = ((self.discount/(1-self.discount)*(1-self.discount**self.distance))*orders[np.repeat(np.arange(num_acts),orders.shape[1]).reshape(num_acts,-1),next_storage_maps[np.arange(num_acts)]-1]).sum(axis=1)
        # costs = delay_costs + self.exchange_cost_weight*exchange_costs
        # rewards = - costs
        # next_states = self.storage_map_to_state(next_storage_maps)
        # return next_states, rewards, delay_costs

    def set_state(self, state):
        storage_map = self.state_to_storage_map(state)
        self._wrapped_env.set_state(storage_map)

    def vec_set_state(self, states):
        storage_maps = self.state_to_storage_map(states)
        self._wrapped_env.vec_set_state(storage_maps)

    def storage_map_to_state(self, storage_maps, p):
        inverse_perm = np.arange(self.num_products)[np.argsort(storage_maps)]
        # GGdist = np.take_along_axis(self.dist_matrix[inverse_perm], np.repeat(
        #     np.expand_dims(inverse_perm, axis=-2), self.num_products, axis=-2), axis=-1)
        # GGdist_upper_tri = GGdist.T[np.triu_indices(self.num_products, k=1)].T
        good_to_exit = self.distance[storage_maps-1]
        # states = np.hstack([p, good_to_exit, GGdist_upper_tri, storage_maps])
        # states = np.hstack([p, good_to_exit, storage_maps])
        states = np.hstack([p, storage_maps])
        return states

    def state_to_storage_map(self, states):
        return (states.T[-self.num_products:].T.astype(int))

    def state_to_p_state(self, states):
        return states.T[:self.num_products*self.num_p_in_states].T

    def sample_states(self, batch_size):
        storage_maps = random_permutation(range(1,self.num_products+1),batch_size)
        ind = np.random.choice(self.p_features_set.shape[0], batch_size, replace = True)
        sampled_p = self.p_features_set[ind]
        p_current = sampled_p[:,:-1,:].reshape((batch_size,-1))
        p_next = sampled_p[:,1:,:].reshape((batch_size,-1))
        return self.storage_map_to_state(storage_maps, p_current), p_current, p_next

    def sample_actions(self, num_acts, all=False):
        if all:
            if not hasattr(self, '_all_actions'):
                self._all_actions = np.array(list(itertools.combinations(range(self.num_products), 2))+[None])
            actions = self._all_actions
        else:
            actions = np.sort(random_choice_noreplace(range(self.num_products), n_sample=2 ,num_draw=num_acts-1),axis=-1)
            actions = np.array(list(map(tuple,actions))+[None])
        return actions


    def random_choice_noreplace(self, l, n_sample, num_draw):
        '''
        l: 1-D array or list
        n_sample: sample size for each draw
        num_draw: number of draws

        Intuition: Randomly generate numbers, get the index of the smallest n_sample number for each row.
        '''
        l = np.array(l)
        return l[np.argpartition(np.random.rand(num_draw,len(l)), n_sample-1,axis=-1)[:,:n_sample]]

    def random_permutation(self, l, n_sample, num_draw):
        # m, n are the number of rows, cols of output
        l = np.array(l)
        return l[np.random.rand(num_draw,len(l)).argsort(axis=-1)[:,:n_sample]]


    def upsample(self, image, scale):
        up_image = np.repeat(image, scale, axis=-2)
        up_image = np.repeat(up_image, scale, axis=-1)
        return up_image

    def downsample(self, image, scale):
        down_image = image.take(np.arange(image.shape[-2], step=scale), axis=-2).take(
            np.arange(image.shape[-1], step=scale), axis=-1)
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
    from envs import ASRSEnv, StaticOrderProcess
    from algos import TruePPredictor
    op = StaticOrderProcess(num_products = 9) 
    base_env = ASRSEnv((3, 3), op)
    true_p = TruePPredictor(base_env, look_back=1000, dynamic=False, init_num_period = 1000, num_p_in_states =1)
    env = DynamicProbEnv(base_env,demand_predictor = true_p, alpha=1, num_p_in_states = 1)
    env.sample_actions(5)
    env.sample_states(5)
    print(env.step(None))

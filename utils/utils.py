import numpy as np
import time
from collections.abc import Iterable
from gym import spaces, Env

class RandomPolicy(object):
    def __init__(self, env):
        self.num_products = env.num_products

    def get_action(self, state):
        action = np.random.choice(self.num_products,2,replace=False)
        return action

class SimpleMaxPolicy(object):
    def __init__(self, env, value_fun, num_acts):
        self.env = env
        self.discount = env.discount
        self.num_products = env.num_products
        self._value_fun = value_fun
        self.num_acts = env.num_actions

    def get_action(self, state):
        if state.ndim ==self.env.obs_dim:
            num_states = 1
            states = np.array([state]*self.num_acts)
        elif state.ndim == self.env.obs_dim +1:
            num_states = state.shape[0]
            states = np.tile(state.T, self.num_acts).T
        else:
            raise NotImplementedError
        actions = self.env.sample_actions(self.num_acts, all=True)
        rep_actions = np.repeat(actions, num_states, axis=0)
        total_rewards = np.zeros(self.num_acts*num_states)
        self.env.vec_set_state(states)
        next_states, rewards, delay_costs = self.env.vec_step(rep_actions)
        total_rewards += rewards + (self.discount)*self._value_fun.get_values(next_states)
        total_rewards = total_rewards.reshape((self.num_acts, num_states))
        if num_states == 1:
            self.env.set_state(state)
        else:
            self.env.vec_set_state(state)
        return actions[np.argmax(total_rewards,axis=0)]

class TabularPolicy(object):
    def __init__(self, env):
        # assert isinstance(env.action_space, spaces.Discrete)
        # assert isinstance(env.observation_space, spaces.Discrete)
        self.num_actions = env.num_actions
        self.num_states = env.num_states
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))

    def get_action(self, state):
        probs = np.array(self._policy[state])
        if probs.ndim == 2:
            probs = probs / np.expand_dims(np.sum(probs, axis=-1), axis=-1)
            s = probs.cumsum(axis=-1)
            r = np.expand_dims(np.random.rand(probs.shape[0]), axis=-1)
            action = (s < r).sum(axis=1)
        elif probs.ndim == 1:
            # idxs = np.random.multinomial(1, probs / np.sum(probs))
            # action = np.argmax(idxs)
            action = np.random.choice(a=self.num_actions, size=1, p=probs/np.sum(probs))[0]
        else:
            raise NotImplementedError
        return action

    def get_probs(self):
        return np.array(self._policy) / np.expand_dims(np.sum(self._policy, axis=-1), axis=-1)

    def update(self, pi):
        assert (pi >= 0).all()
        assert pi.shape[0] == self.num_states
        if pi.ndim == 1:
            self._policy[:, :] = 0
            self._policy[range(self.num_states), pi] = 1.
        elif pi.ndim == 2:
            self._policy = pi
        else:
            raise TypeError


class TabularValueFun(object):
    def __init__(self, env):
        self.num_states = env.num_states
        self._value_fun = np.zeros(shape=(self.num_states,))

    def get_values(self, states=None):
        if states is None:
            return self._value_fun
        else:
            return self._value_fun[states]

    def update(self, values):
        self._value_fun = values

class LookAheadPolicy(object):
    """
    Look ahead policy

    -- UTILS VARIABLES FOR RUNNING THE CODE --
    * look_ahead_type (str): Type of look ahead policy to use

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * self.num_elites (int): number of best actions to pick for the cross-entropy method

    * self.value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

    * self.get_returns_state(state): It is the same that you implemented in the previous part
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 look_ahead_type='tabular',
                 num_acts=20,
                 cem_itrs=10,
                 precent_elites=0.25,
                 ):
        self.env = env
        self.discount = self.env.discount
        self._value_fun = value_fun
        self.horizon = horizon
        self.num_acts = num_acts
        self.cem_itrs = cem_itrs
        self.num_elites = int(num_acts * precent_elites)
        assert self.num_elites > 0
        self.look_ahead_type = look_ahead_type

    def get_action(self, state):
        if self.look_ahead_type == 'tabular':
            action = self.get_action_tabular(state)
        elif self.look_ahead_type == 'rs':
            action = self.get_action_rs(state)
        elif self.look_ahead_type == 'cem':
            action = self.get_action_cem(state)
        else:
            raise NotImplementedError
        return action

    def get_action_cem(self, state):
        """
        Do lookahead in the continous and discrete case with the cross-entropy method..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts
        """ INSERT YOUR CODE HERE"""
        if isinstance(self.env.action_space, spaces.Discrete):
            n_act = self.env.action_space.n
            freq = np.ones((self.horizon,n_act))
            for _ in range(self.cem_itrs):
                actions = np.empty((self.horizon,num_acts),dtype=int)
                for t in range(self.horizon):
                    actions[t] = np.argmax(np.random.multinomial(1,freq[t]/freq[t].sum(),size=num_acts),axis=1)
                returns = self.get_returns(state, actions)
                elite_actions_ind = np.argpartition(returns, -self.num_elites)[-self.num_elites:]
                elite_actions = actions[:,elite_actions_ind]
                freq += np.apply_along_axis(np.bincount,axis=1,arr=elite_actions,minlength=n_act)
            best_action = np.argmax(freq[0])
        else:
            act_low, act_high = self.env.action_space.low, self.env.action_space.high
            act_dim = len(act_low)
            std = np.eye(act_dim)*(act_high - act_low)/4
            mu = np.zeros((self.horizon,act_dim))
            for _ in range(self.cem_itrs):
                actions = np.empty((self.horizon,num_acts,act_dim))
                for t in range(self.horizon):
                    actions[t] = np.random.multivariate_normal(mu[t],np.matmul(std.T,std),num_acts)
                for i in range(act_dim):
                    actions[actions<act_low[i]] = act_low[i]
                    actions[actions>act_high[i]] = act_high[i]
                returns = self.get_returns(state, actions)
                elite_actions_ind = np.argpartition(returns, -self.num_elites)[-self.num_elites:]
                elite_actions = actions[:,elite_actions_ind]
                mu = np.mean(elite_actions,axis=1)
            best_action = mu[0,:]
            """ Your code ends here """
        return best_action

    def get_action_rs(self, state):
        """
        Do lookahead in the continous and discrete case with random shooting..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts
        """ INSERT YOUR CODE HERE """
        # if isinstance(self.env.action_space, spaces.Discrete):
        n_act = self.env.num_actions
        if state.ndim == self.env.obs_dim:
            actions = np.random.randint(0,n_act,size=(self.horizon,num_acts))
            returns = self.get_returns(state, actions)
            best_action = actions[0, np.argmax(returns)]
        elif state.ndim == self.env.obs_dim+1:
            actions = np.random.randint(0,n_act,size=(self.horizon*state.shape[0],num_acts))
            returns = self.get_returns(state, actions)
            best_action = actions[0, np.argmax(returns)]
        else:
            raise NotImplementedError
            # actions_each_t = np.random.randint(0,n_act,size=(self.horizon,num_acts))
            # actions = np.array(list(itertools.product(*[np.unique(row) for row in actions_each_t]))).T
        # else:
        #     assert num_acts is not None
        #     act_low, act_high = self.env.action_space.low, self.env.action_space.high
        #     actions = np.random.uniform(act_low, act_high, size=(self.horizon, num_acts, len(act_low)))
        return self.env.get_action_from_id(best_action)

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        """ INSERT YOUR CODE HERE"""
        H = self.horizon
        n_total_actions = actions.shape[1]
        self.env.vec_set_state(np.array([state]*n_total_actions))
        returns = np.zeros(n_total_actions)
        not_done_status = np.ones(n_total_actions, dtype=bool)
        for t in range(H):
            id_next_s, costs, delay_costs = self.env.vec_step(actions[t,:])
            returns += (self.discount**t)*costs
            self.env.vec_set_state(id_next_s)
        returns += (self.discount**H)*self._value_fun.get_values(id_next_s)
        return returns



class SparseArray(object):
    def __init__(self, obs_n, act_n, mode, obs_dims=None):
        if mode == 'nn':
            next_obs_n = 1
        elif mode == 'linear':
            assert obs_dims is not None
            next_obs_n = int(2 ** obs_dims)
        else:
            raise NotImplementedError
        self._obs_n = obs_n
        self._act_n = act_n
        self._mode = mode
        self._obs_dims = obs_dims
        self._values = np.zeros((obs_n, act_n, next_obs_n), dtype=np.float32)
        self._idxs = np.zeros((obs_n, act_n, next_obs_n), dtype=int)
        self._fill = np.zeros((obs_n, act_n), dtype=int)

    def __mul__(self, other):
        if isinstance(other, SparseArray):
            assert (self._idxs == other._idxs).all(), "other does not have the same sparsity"
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values * other._values

        elif isinstance(other, np.ndarray):
            assert other.shape == (1, 1, self._obs_n)
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values * other[self._idxs]

        else:
            raise NotImplementedError

        return result

    def __add__(self, other):
        if isinstance(other, SparseArray):
            assert (self._idxs == other._idxs).all(), "other does not have the same sparsity"
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values + other._values

        elif isinstance(other, np.ndarray):
            assert other.shape == (self._obs_n,)
            result = SparseArray(self._obs_n, self._act_n, self._mode, self._obs_dims)
            result._idxs = self._idxs
            result._values = self._values + other[self._idxs]

        else:
            raise NotImplementedError

        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _inputs = tuple()
        for inp in inputs:
            if isinstance(inp, SparseArray):
                _inputs += (inp._values,)
            else:
                _inputs += (inp,)
        return getattr(ufunc, method)(*_inputs, **kwargs)

    def sum(self, *args, **kwargs):
        return self._values.sum(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self._values.max(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self._values.reshape(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self._values.transpose(*args, **kwargs)

    def __setitem__(self, key, value):
        if type(key) is not tuple:
            self._values[key] = value
        elif len(key) == 2:
            obs, act = key
            self._values[obs, act] = value
        else:
            obs, act, n_obs = key
            if self._mode == 'nn':
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    assert (value.shape[0] == 1 or value.shape[1] == 1)
                    value = value.reshape(-1)
                if isinstance(obs, np.ndarray) and obs.ndim == 2:
                    assert (obs.shape[0] == 1 or obs.shape[1] == 1)
                    obs = obs.reshape(-1)
                if isinstance(act, np.ndarray) and act.ndim == 2:
                    assert (act.shape[0] == 1 or act.shape[1] == 1)
                    act = act.reshape(-1)
                if isinstance(n_obs, np.ndarray) and n_obs.ndim == 2:
                    assert (n_obs.shape[0] == 1 or n_obs.shape[1] == 1)
                    n_obs = n_obs.reshape(-1)

                self._values[obs, act, 0] = value
                self._idxs[obs, act, 0] = n_obs

            elif self._mode == 'linear':
                if isinstance(n_obs, np.ndarray) and n_obs.ndim == 2:
                    assert n_obs.shape[-1] == int(2 ** self._obs_dims)
                    if value.ndim == 1:
                        self._values[obs, act, :] = np.expand_dims(value, axis=-1)
                    else:
                        self._values[obs, act, :] = value
                    self._idxs[obs, act, :] = n_obs
                else:
                    self._values[obs, act, self._fill[obs, act]] = value
                    self._idxs[obs, act, self._fill[obs, act]] = n_obs
                    self._fill[obs, act] += 1

    def __getitem__(self, key):
        if type(key) is not tuple:
            return self._values[key]
        elif len(key) == 2:
            obs, act = key
            return self._values[obs, act]
        else:
            obs, act, n_obs = key
            if self._mode == 'nn':
                assert (n_obs == self._idxs[obs, act, 0]).all()
                return self._values[obs, act, 0]

            elif self._mode == 'linear':
                assert (n_obs == self._idxs[obs, act, self._fill[obs, act]]).all()
                return self._values[obs, act, self._fill[obs, act]]


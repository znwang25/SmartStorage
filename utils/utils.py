import numpy as np
import time
from collections.abc import Iterable
from gym import spaces, Env

class RandomPolicy(object):
    def __init__(self, env):
        self.num_products = env.num_products

    def get_action_vec(self, states):
        n = states.shape[0]
        actions = []
        for i in range(n):
            actions.append(np.random.choice(self.num_products,2,replace=False))
        return actions

    def get_action(self, state):
        action = np.random.choice(self.num_products,2,replace=False)
        return action

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


class TabularPolicy(object):
    def __init__(self, env):
        # assert isinstance(env.action_space, spaces.Discrete)
        # assert isinstance(env.observation_space, spaces.Discrete)
        self.act_dim = env.action_dim
        self.num_states = env.num_states
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.act_dim))

    def get_action(self, state):
        probs = np.array(self._policy[state])
        if probs.ndim == 2:
            probs = probs / np.expand_dims(np.sum(probs, axis=-1), axis=-1)
            s = probs.cumsum(axis=-1)
            r = np.expand_dims(np.random.rand(probs.shape[0]), axis=-1)
            action = (s < r).sum(axis=1)
        elif probs.ndim == 1:
            idxs = np.random.multinomial(1, probs / np.sum(probs))
            action = np.argmax(idxs)
        else:
            raise NotImplementedError
        return action

    def get_probs(self):
        return np.array(self._policy) / np.expand_dims(np.sum(self._policy, axis=-1), axis=-1)

    def update(self, actions):
        assert (actions >= 0).all()
        assert actions.shape[0] == self.num_states
        if actions.ndim == 1:
            self._policy[:, :] = 0
            self._policy[range(self.num_states), actions] = 1.
        elif actions.ndim == 2:
            self._policy = actions
        else:
            raise TypeError


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


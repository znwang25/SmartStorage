import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import autograd.numpy as np
from collections import OrderedDict

class CNNValueFun(object):
    def __init__(self, env, activation='relu'):
        self._env = env
        input_shape = env.state_shape
        self._build((*input_shape,1), activation)
    
    def _build(self, input_shape, activation='relu', *args, **kwargs):
        #create model
        model = Sequential()
        #add model layers
        model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(Conv2D(5, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1,activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def get_values(self, states):
        states = np.expand_dims(states, axis=-1)
        return self.model.predict(states).reshape(-1)

    def update(self, states, V_bar):
        states = np.expand_dims(states, axis=-1)
        # loss = self.model.train_on_batch(states, V_bar,reset_metrics=False)
        loss = self.model.fit(states, V_bar)
        print(loss)


class MLPValueFun(object):
    _activations = {
        'tanh': np.tanh,
        None: lambda x: x,
        'relu': lambda x: np.maximum(x, 0)
    }

    def __init__(self, env, hidden_sizes=(256, 256), activation='relu'):
        self._env = env
        self._params = dict()
        self._build(hidden_sizes, activation)

    def _build(self, hidden_sizes=(256, 256), activation='relu', *args, **kwargs):
        self._activation = self._activations[activation]
        self._hidden_sizes = hidden_sizes
        prev_size = self._env.observation_space.shape[0]
        for i, hidden_size in enumerate(hidden_sizes):
            W = np.random.normal(loc=0, scale=1/prev_size, size=(hidden_size, prev_size))
            b = np.zeros((hidden_size,))

            self._params['W_%d' % i] = W
            self._params['b_%d' % i] = b

            prev_size = hidden_size

        W = np.random.normal(loc=0, scale=1/prev_size, size=(1, prev_size))
        b = np.zeros((1,))
        self._params['W_out'] = W
        self._params['b_out'] = b

    def get_values(self, states, params=None):
        params = self._params if params is None else params
        x = states
        for i, hidden_size in enumerate(self._hidden_sizes):
            x = np.dot(params['W_%d' % i], x.T).T + params['b_%d' % i]
            x = self._activation(x)
        values = np.dot(params['W_out'], x.T).T + params['b_out']
        return values[:, 0]

    def update(self, params):
        assert set(params.keys()) == set(self._params.keys())
        self._params = params


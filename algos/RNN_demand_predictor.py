import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

class RNNDemandPredictor(object):
    def __init__(self, env, look_back=1000, init_num_period = 10000, epochs= 2):
        self._env = env
        self.look_back = look_back
        self.epochs = epochs
        self.num_products = env.num_products
        self.init_num_period = init_num_period
        order_sequence, _ = env.get_order_sequence(num_period=init_num_period)
        self.order_sequence_buffer = order_sequence
        self._build()
        self.update(self.order_sequence_buffer)
        self.buffer_p_sequence_hat = self.get_predicted_p(self.order_sequence_buffer, preprocess= True)

    def _build(self):
        model = Sequential()
        model.add(LSTM(units=50, input_shape=(self.look_back, self.num_products)))
        # model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, num_products)))
        model.add(Dropout(0.2))
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=50, return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=50))
        # model.add(Dropout(0.2))
        model.add(Dense(units = self.num_products, activation="sigmoid"))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        self.model = model

    def _preprocess_data(self, order_sequence):
        features_set = self.sliding_window(order_sequence, self.look_back)
        labels = order_sequence[self.look_back:]
        return features_set, labels

    def sliding_window(self, arr, window_len): 
        # Took from https://stackoverflow.com/a/43185821/8673150
        # INPUTS :
        # arr is array
        # window_len is length of array along axis=0 to be cut for forming each subarray

        # Length of 3D output array along its axis=0, we will omit the last window
        num_entries = arr.shape[0] - window_len

        # Store shape and strides info
        m,n = arr.shape
        s0,s1 = arr.strides

        # Finally use strides to get the 3D array view
        return np.lib.stride_tricks.as_strided(arr, shape=(num_entries,window_len,n), strides=(s0,s0,s1))

    def get_predicted_p(self, features_set, preprocess = False):
        if preprocess:
            features_set, _ = self._preprocess_data(features_set)
        elif features_set.ndim == 2:
            features_set = np.expand_dims(features_set,axis=0)
        return self.model.predict(features_set)

    def update(self, order_sequence):
        features_set, labels = self._preprocess_data(order_sequence)
        loss = self.model.fit(features_set, labels, epochs = self.epochs, batch_size = 200)
    
    def test_performance_plot(self, test_num_period, save_to = None):
        test_order_sequence, test_p_sequence = self._env.get_order_sequence(num_period=test_num_period)
        test_features_set, _ = self._preprocess_data(test_order_sequence)
        test_p_sequence_hat = self.get_predicted_p(test_features_set)
        test_p_sequence = test_p_sequence[self.look_back:]
        plt.clf()
        plt.figure(figsize=(50,10))
        for i in range(self.num_products):
            color = np.random.rand(3,)
            print(color)
            plt.plot(test_p_sequence[:,i], c=color, linestyle='-')  
            plt.plot(test_p_sequence_hat[:,i], c=color, linestyle=':')
        plt.xlabel("t")
        plt.ylabel("p")
        if save_to:
            plt.savefig('%s/rnn_performance_plot.png' % (save_to))
        else:
            plt.show()

    

class TruePPredictor(object):
    def __init__(self, env, dynamic= False, look_back = 1000, init_num_period = 10000,num_p_in_states = 10):
        self._env = env
        self.look_back= look_back
        self.dynamic = dynamic
        if dynamic:
            _, true_p_sequence = env.get_order_sequence(num_period=init_num_period)
        else:
            _, true_p_sequence = env.get_order_sequence(num_period=num_p_in_states+2)
        self.buffer_p_sequence_hat = true_p_sequence
        self.num_p_in_states = num_p_in_states


    def sliding_window(self, arr, window_len): 
            # Took from https://stackoverflow.com/a/43185821/8673150
            # INPUTS :
            # arr is array
            # window_len is length of array along axis=0 to be cut for forming each subarray

            # Length of 3D output array along its axis=0, we will omit the last window
            num_entries = arr.shape[0] - window_len

            # Store shape and strides info
            m,n = arr.shape
            s0,s1 = arr.strides

            # Finally use strides to get the 3D array view
            return np.lib.stride_tricks.as_strided(arr, shape=(num_entries,window_len,n), strides=(s0,s0,s1))

    def get_predicted_p(self, features_set,preprocess = False):
        if self.dynamic:
            self.buffer_p_sequence_hat = np.vstack([self.buffer_p_sequence_hat, self._env.dist_param])[1:]
        if preprocess:
            return self.buffer_p_sequence_hat[-self.num_p_in_states:]
        return self.buffer_p_sequence_hat[-1]
    

# test_order_sequence, test_p_sequence = a.get_order_sequence(num_period=2000)
# test_features_set, _ = rnn._preprocess_data(test_order_sequence)
# test_p_sequence_hat = rnn.get_predicted_p(test_features_set)
# test_p_sequence = test_p_sequence[rnn.look_back:]
# for i in range(3):
#     color = np.random.rand(3,)
#     print(color)
#     plt.plot(test_p_sequence[:,i], c=color, linestyle='-')  
#     # plt.plot(test_p_sequence_hat[:,i], c=color, linestyle=':')  
# plt.show()
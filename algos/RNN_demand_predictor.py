import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

class RNNPredictor(object):
    def __init__(self, env, look_back=1000, init_num_period = 10000):
        self._env = env
        self.look_back = look_back
        self.num_products = env.num_products
        order_sequence, _ = env.get_order_sequence(num_period=init_num_period)
        self._build()
        self.update(order_sequence)

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

    def get_predicted_p(self, features_set):
        return self.model.predict(features_set)

    def update(self, order_sequence):
        features_set, labels = self._preprocess_data(order_sequence)
        loss = self.model.fit(features_set, labels, epochs = 2, batch_size = 200)
    
    def test_performance_plot(self, test_num_period):
        test_order_sequence, test_p_sequence = self._env.get_order_sequence(num_period=test_num_period)
        test_features_set, _ = self._preprocess_data(test_order_sequence)
        test_p_sequence_hat = self.get_predicted_p(test_features_set)
        test_p_sequence = test_p_sequence[self.look_back:]
        plt.clf()
        for i in range(self.num_products):
            color = np.random.rand(3,)
            print(color)
            plt.plot(test_p_sequence[:,i], c=color, linestyle='-')  
            plt.plot(test_p_sequence_hat[:,i], c=color, linestyle=':')  
        plt.show()
    

test_order_sequence, test_p_sequence = a.get_order_sequence(num_period=2000)
test_features_set, _ = rnn._preprocess_data(test_order_sequence)
test_p_sequence_hat = rnn.get_predicted_p(test_features_set)
test_p_sequence = test_p_sequence[rnn.look_back:]
for i in range(3):
    color = np.random.rand(3,)
    print(color)
    plt.plot(test_p_sequence[:,i], c=color, linestyle='-')  
    # plt.plot(test_p_sequence_hat[:,i], c=color, linestyle=':')  
plt.show()
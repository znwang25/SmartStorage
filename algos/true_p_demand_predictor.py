
import numpy as np

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
            self.buffer_p_sequence_hat = np.vstack([self.buffer_p_sequence_hat, self._env.order_process.dist_param])[1:]
        if preprocess:
            return self.buffer_p_sequence_hat[-self.num_p_in_states:]
        return self.buffer_p_sequence_hat[-1]
    
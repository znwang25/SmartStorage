import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

num_product = 10
len_order_sequence = 10000

order_sequence = np.random.binomial(1, 0.5, (len_order_sequence, num_product))
look_back = 1000
num_entries = order_sequence.shape[0] - look_back

# features_set = np.zeros((num_entries, look_back, num_product))
# labels = np.zeros((num_entries, num_product))
# for i in range(num_entries):
#     features_set[i] = order_sequence[i:i+look_back]
#     labels[i] = order_sequence[i+look_back]

# This function is so much faster than the for loop, not even on the same magnitude.
def sliding_window(arr, window_len): 
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

features_set = sliding_window(order_sequence, look_back)
labels = order_sequence[look_back:]

model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, num_product)))
# model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, num_product)))
model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
model.add(Dense(units = num_product, activation="sigmoid"))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

model.fit(features_set, labels, epochs = 50, batch_size = 100)

a = np.random.binomial(1, 0.5, (len_order_sequence, num_product))
test_features = sliding_window(a, look_back)

predictions = model.predict(test_features)
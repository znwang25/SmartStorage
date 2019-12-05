num_product = 10
len_order_sequence = 2000

order_sequence = np.random.binomial(1, 0.5, (len_order_sequence, num_product))
look_back = 1000
num_entries = order_sequence.shape[0] - look_back


features_set = np.zeros((num_entries, look_back, num_product))
labels = np.zeros(num_entries, num_product)

for i in range(num_entries):
    features_set[i] = order_sequence[i:i+look_back]
    labels[i] = lables[i+look_back:i+look_back+num_product]


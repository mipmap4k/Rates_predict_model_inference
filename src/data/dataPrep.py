import numpy as np

def create_sequences(data, seq_length):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), min_val, max_val

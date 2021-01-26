import numpy as np

data = np.load('../sketchrnn_chair.npz', encoding='latin1', allow_pickle=True)
print(np.shape(data))
print(data['valid'])
print(np.shape(data['valid']))
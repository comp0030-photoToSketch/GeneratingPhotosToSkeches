import numpy as np
from encoder import *
from dataLoader import *


dataLoader = DataLoader('../sketchrnn_chair.npz')
data = dataLoader.load_npz()
for strokedata in data['valid']:
    # print(np.shape(strokedata)[0])
    print(strokedata)
# print(data['valid'])

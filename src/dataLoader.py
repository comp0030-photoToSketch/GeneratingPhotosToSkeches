import numpy as np


class DataLoader:
    def __init__(self, file):
        self.file = file

    def load_npz(self):
        data = np.load(self.file, encoding='latin1', allow_pickle=True)
        return data

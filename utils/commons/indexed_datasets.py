# Based on https://github.com/NATSpeech/NATSpeech
import numpy as np
import os
import pickle

from copy import deepcopy


class IndexedDataset:
    def __init__(self, path: str, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()["offsets"]
        self.data_file = open(f"{path}.data", "rb", buffering=-1)
        self.cache = []
        self.num_cache = num_cache
    
    def check_index(self, i: int):
        if i < 0 or i > len(self.data_offsets) - 1:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1
    

class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.read_type = "wb"
        if os.path.exists(f"{path}.data"):
            self.read_type = "ab"
        self.out_file = open(f"{path}.data", self.read_type)
        self.byte_offsets = [0]
        if os.path.exists(f"{self.path}.idx"):
            self.byte_offsets = np.load(f"{path}.idx", allow_pickle=True).item()["offsets"]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", "wb"), {"offsets": self.byte_offsets})

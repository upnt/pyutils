import numpy as np


class nddict:
    def __init__(self, keys, key_dict, func=None):
        self._keys = set()
        self._key_dict = {}
        self._array = None
        self._func = lambda: None if func is None else func

    def add_key(self, label, key_list):
        if label in self._key_dict.keys():
            raise ValueError(f"{label} is already exist")
        for key in key_list:
            if key in self._keys:
                raise ValueError(f"{key} is already exist")
        self._key_dict[label] = key_list
        self._keys |= set(key_list)
        if self._array is None:
            self._array = [None for _ in key_list]
        else:
            self._array = np.stack([np.full_like(self._array, self._func()) for _ in key_list], 0)

    def __getitem__(self, key):
        for label, key_list in self._key_dict.items():
            if key in key_list:
                array_key = tuple(0 if i == label else slice(None, None, None) for i in self._key_dict)
                return self._array[array_key]
        raise ValueError(f"{key} is not in {self}")

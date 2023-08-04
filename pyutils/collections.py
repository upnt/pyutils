import numpy as np


class nddict:
    def __init__(self, keys, values, /, func=None):
        self._keys = keys
        self._values = values
        self._func = lambda: None if func is None else func
    
    @staticmethod
    def from_keys(keys) -> "nddict":
        result = nddict(None, None)
        for key in keys:
            result._add_key(key)
        return result

    def _add_key(self, key_list):
        if self._keys is None:
            self._keys = [key_list]
        else:
            self._keys.append(key_list)

        if self._values is None:
            self._values = np.array([None for _ in key_list])
        else:
            self._values = np.stack([np.full_like(self._values, self._func()) for _ in key_list], axis=-1)

    def __getitem__(self, key):
        flag = False
        for i, key_list in enumerate(self._keys):
            if key in key_list:
                major = i
                miner = key_list.index(key)
                flag = True
                break
        if not flag:
            raise KeyError(f"{key} is not in {self._keys}")
        keys = self._keys.copy()
        del keys[major]
        index = tuple(miner if i == major else slice(None, None, None) for i in range(self._values.ndim))
        values = self._values[index]
        if len(keys) == 0:
            return values
        else:
            return nddict(keys, values, func=self._func)
    
    def __setitem__(self, key, value):
        if hasattr(key, '__iter__'):
            key = tuple(key)
        else:
            key = (key,)
        if len(key) != len(self._keys):
                raise ValueError(f"Dimentions of {self._values} is smaller than {len(key)}")
        index = []
        for k, key_list in zip(key, self._keys):
            if isinstance(k, slice):
                index.append(k)
            else:
                if k not in key_list:
                    raise KeyError(f"{k} is not in {key_list}")
                index.append(key_list.index(k))
        self._values[tuple(index)] = value

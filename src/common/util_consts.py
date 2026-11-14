import numpy as np

import util_io


class consts:
    def __init__(self, data):
        self._data = dict(data)
        self.reads = set()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f'no constant "{name}" found; available ones are: {sorted(self._data)}')
        self.reads.add(name)
        return value

    def to_string(self):
        return '\n'.join([f'{k}: {v}' for k, v in self._data.items()])

    def dump(self, path):

        # --! json does not support numpy arrays, so convert them to lists
        dumped = {}
        for k, v in self._data.items():
            if isinstance(v, np.ndarray):
                dumped[k] = v.tolist()
            else:
                dumped[k] = v
        util_io.write_json(dumped, path)


def load(path):

    # --! read .json file
    loaded = util_io.read_json(path)

    # --! convert lists into numpy arrays to set the correct data type
    array_corrected = {}
    for name, value in loaded.items():
        if isinstance(value, list):
            array_corrected[name] = np.asarray(value, dtype=np.float32)
        else:
            array_corrected[name] = value

    # --! return consts object
    return consts(array_corrected)


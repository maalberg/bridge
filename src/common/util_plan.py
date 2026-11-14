import os
import numpy as np

import util_io
import util_dpu
import util_consts


class plan:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f'no field "{name}" found in plan')
        return value

    def __contains__(self, key):
        return key in self._data

    def to_consts(self, state, source='state'):
        buffers = {}
        for name, shape in self._data['buffers'].items():

            # --! make sure that this buffer with this name is in state
            if name not in state:
                raise SystemExit(f'[error]: plan lists buffer {name}, but {source} has no such key')

            value = state[name].detach().cpu().numpy().astype(np.float32)
            if list(value.shape) != list(shape):
                raise SystemExit(f'[error]: buffer {name} has shape {list(value.shape)}, but plan says {list(shape)}')

            buffers[name] = value.tolist()

        # --! add buffers to config attributes to form a single constants dictionary
        consts = dict(self._data['config_attributes'])
        for name, value in buffers.items():
            consts[name] = np.asarray(value, dtype=np.float32)

        return util_consts.consts(consts)

    def to_dpus(self, xmodels):
        """
        Given ``xmodels`` path, creates dpus from this plan.
        """
        return {
            name: util_dpu.stage(
                os.path.join(xmodels, name + '.xmodel'),
                in_shape=i,
                out_shape=o) for name, (i, o) in self.get_stage_rows().items()}

    def get_stage_rows(self):
        if 'trace' not in self._data:
            raise SystemExit('[error]: plan has no behaviour trace; run analyze_model.py with --data')

        rows = {}
        for name, t in self._data['trace'].items():
            if t['ncalls'] == 0:
                continue
            if len(t['in_shapes']) != 1 or len(t['out_shapes']) != 1:
                raise SystemExit(
                    f'[error]: stage {name} sees several shapes ({t["in_shapes"]} -> {t["out_shapes"]}); a DPU stage needs fixed row shapes')

            rows[name] = (tuple(t['in_shapes'][0][1:]), tuple(t['out_shapes'][0][1:]))
        return rows


def load(path):
    """
    Loads model plan from a .json file located at ``path``.

    Required parameters:
        path        string, path to a model plan saved in a .json file

    Returns:
        plan        plan object
    """
    
    return plan(util_io.read_json(path))


import torch

from util_io import read_config, read_trajectories, import_module
from util_model import build_model, check_weights, make_input_sets


def load_session(ns, strict=True):
    model, construction = build_model(import_module(ns.model, 'model', ['model']), read_config(ns.config))

    weights_check = {}
    if ns.weights:
        weights_check = check_weights(model, ns.weights)

        # --! a model that silently runs with some of its weights not loaded gives plausible but wrong results
        if strict and (weights_check['missing_keys'] or weights_check['unexpected_keys']):
            raise SystemExit(
                f'[error]: weights {ns.weights} do not match model: '
                f'missing keys {weights_check["missing_keys"]}, unexpected keys {weights_check["unexpected_keys"]}')
    model.eval()

    input_sets = []
    trajectories = None
    if ns.data:
        trajectories = torch.from_numpy(read_trajectories(ns.data, ns.traj_nsample))
        input_sets = make_input_sets(model, trajectories)

    return model, construction, weights_check, trajectories, input_sets


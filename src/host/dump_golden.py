"""
Host-side script to save stage-specific golden vectors for evaluation.
"""

import os
import torch
import numpy as np

from pytorch_nndct.apis import torch_quantizer

from util import load_session
from util_io import read_arguments
from util_model import get_stage_names, capture_stage_inputs


def main():

    # --! read arguments with extra calibration-specific arguments and options
    args = {
        'stage': {
            'help'    : 'name of child module whose inputs to dump, e.g. encoder'
        }
    }
    options = {
        '--traj-id': {
            'type'    : int,
            'help'    : 'which trajectory supplies the rows, default=0',
            'default' : 0
        },
        '--out-golden': {
            'help'    : 'path of the saved .npz file, default=golden_<stage>.npz'
        }
    }
    ns = read_arguments(description='record golden vectors of one quantized stage', extra_args=args, extra_options=options)

    # --! we do not want to change the properties of base options, so impose extra constraints on them at run time
    if not ns.data or not ns.traj_nsample or not ns.weights:
        raise SystemExit('[error]: for golden record, --data, --traj-nsample, and --weights are required')

    # --! load current session
    model, _, _, _, input_sets = load_session(ns)

    # --! check that targeted stage exists in given model
    if ns.stage not in get_stage_names(model):
        raise SystemExit(f'[error]: model has no child module {ns.stage}; stages are {get_stage_names(model)}')

    # --! check trajectory number
    if not 0 <= ns.traj_id < len(input_sets):
        raise SystemExit(f'[error]: --traj-id must be in 0..{len(input_sets) - 1}')

    # --! the rows this stage receives inside the float pipeline
    x = capture_stage_inputs(model, ns.stage, [input_sets[ns.traj_id]])
    print(
        f'[note]: for stage {ns.stage}, reading {x.shape[0]} rows of shape {tuple(x.shape[1:])} from trajectory {ns.traj_id}')

    # --! extract this stage's submodule
    stage = getattr(model, ns.stage)

    # --! float reference first: creating quantizer may modify float module
    with torch.no_grad():
        y_float = stage(x)

    # --! instantiate quantizer in test mode (we assume that calibration has already been done)
    quantizer = torch_quantizer(
        'test',
        stage,
        (x[:1].clone(),),
        output_dir=os.path.join(ns.out_quant, ns.stage),
        device=torch.device('cpu'))

    # --! get quantized model from quantizer
    quant_model = quantizer.quant_model
    quant_model.eval()

    # --! one row at a time, as board does
    with torch.no_grad():
        y_quant = torch.cat([quant_model(x[i:i + 1]) for i in range(x.shape[0])], dim=0)

    # --! save data
    out_file = ns.out_golden or f'golden_{ns.stage}.npz'
    np.savez(
        out_file,
        x=x.detach().cpu().numpy().astype(np.float32),
        y_float=y_float.detach().cpu().numpy().astype(np.float32),
        y_quant=y_quant.detach().cpu().numpy().astype(np.float32))

    print(f'[note]: wrote {out_file} (x {tuple(x.shape)} -> y {tuple(y_quant.shape)})')


if __name__ == '__main__':
    main()


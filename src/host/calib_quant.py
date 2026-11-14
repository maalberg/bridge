
import os

import torch
from pytorch_nndct.apis import torch_quantizer

from util import load_session
from util_io import read_arguments
from util_model import get_stage_names, capture_stage_inputs


def main():

    # --! read arguments with extra calibration-specific arguments and options
    args = {
        'stage': {
            'help'    : 'name of child module to calibrate, e.g. encoder'
        }
    }
    ns = read_arguments(description='calibrate one stage of a composite model', extra_args=args)

    # --! we do not want to change the properties of base options, so impose extra constraints on them at run time
    if not ns.data or not ns.traj_nsample or not ns.weights:
        raise SystemExit('[error]: for calibration, --data, --traj-nsample, and --weights are required')

    # --! load current session
    model, _, weights_check, trajectories, input_sets = load_session(ns)

    # --! check that targeted stage exists in given model
    if ns.stage not in get_stage_names(model):
        raise SystemExit(f'[error]: model has no child module {ns.stage}; stages are {get_stage_names(model)}')

    # --! report on data trajectories read
    print(f'[note]: read {trajectories.shape[0]} trajectories of shape {tuple(trajectories.shape[1:])} from {ns.data}')

    # --! run model stage on input sets to capture stage inputs
    captured = capture_stage_inputs(model, ns.stage, input_sets)
    print(f'[note]: stage {ns.stage} received {captured.shape[0]} samples of shape {tuple(captured.shape[1:])}')

    # --! instantiate quantizer
    #
    # --! as input we pass batch-1 data sample, i.e. first element [:1] from stage inputs captured above
    out_dir = os.path.join(ns.out_quant, ns.stage)
    quantizer = torch_quantizer(
        'calib',
        getattr(model, ns.stage),
        (captured[:1].clone(),),
        output_dir=out_dir,
        device=torch.device('cpu'))

    # --! prepare quantized model
    quant_model = quantizer.quant_model
    quant_model.eval()

    # --! calibrate stage using captured stage inputs
    print(f'[note]: calibrating stage {ns.stage}...')
    with torch.no_grad():

        # --! expose quantized model to all available stage inputs
        for i in range(captured.shape[0]):
            quant_model(captured[i:i + 1])

    # --! export quantization configuration
    quantizer.export_quant_config()
    print(f'[note]: quantization config of stage {ns.stage} written to {out_dir}')


if __name__ == '__main__':
    main()


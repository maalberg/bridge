
import glob
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
            'help'    : 'name of child module to export, e.g. encoder'
        }
    }
    ns = read_arguments(description='export one quantized stage as an xmodel', extra_args=args)

    # --! we do not want to change the properties of base options, so impose extra constraints on them at run time
    if not ns.data or not ns.traj_nsample or not ns.weights:
        raise SystemExit('[error]: for exporting, --data, --traj-nsample, and --weights are required')

    # --! load current session
    model, _, _, _, input_sets = load_session(ns)

    if ns.stage not in get_stage_names(model):
        raise SystemExit(f'[error]: model has no child module {ns.stage}; stages are {get_stage_names(model)}')

    # --! one sample is enough: exporter needs a single test-mode forward with batch size 1
    captured = capture_stage_inputs(model, ns.stage, input_sets[:1])[:1].clone()

    # --! create quantizer in test mode
    out_dir = os.path.join(ns.out_quant, ns.stage)
    quantizer = torch_quantizer(
        'test',
        getattr(model, ns.stage),
        (captured,),
        output_dir=out_dir,
        device=torch.device('cpu'))
    quant_model = quantizer.quant_model
    quant_model.eval()

    # --! test quantized model stage on captured data
    with torch.no_grad():
        quant_model(captured)

    quantizer.export_xmodel(output_dir=out_dir)

    # --! report location of exported file(s)
    files = sorted(glob.glob(os.path.join(out_dir, '*.xmodel')))
    print(f'[note]: stage {ns.stage} exported: {files}')

    # --! double-check file path
    expected = f'{type(getattr(model, ns.stage)).__name__}_int.xmodel'
    if not any(os.path.basename(f) == expected for f in files):
        print(f'[warn]: expected a file named {expected} in {out_dir}')


if __name__ == '__main__':
    main()


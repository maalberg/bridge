"""
Exports model constants. The constants are exported on the host to be then used on the board.
"""

import torch

import util_plan
import util_io


def main():

    # --! read arguments with extra export-specific arguments and options; no base arguments and options
    args = {
        'plan': {
            'help'    : 'model_plan.json from analyze_model.py'
        },
        'weights': {
            'help'    : 'state_dict (a .pth file) of whole composite model'
        }
    }
    options = {
        '--out-consts': {
            'help'    : 'path to a .json file where constants will be saved',
            'default' : 'model_consts.json'
        }
    }
    ns = util_io.read_arguments(
        description='export buffers and config values for board code',
        base_used=False,
        extra_args=args, extra_options=options)

    # --! given model state with its values, extract constants from model plan
    plan = util_plan.load(ns.plan)
    consts = plan.to_consts(torch.load(ns.weights, map_location='cpu'), source=ns.weights)

    # --! write extracted constants into a .json file
    consts.dump(ns.out_consts)

    print(f'[note]: wrote file {ns.out_consts} with:\n{consts.to_string()}')


if __name__ == '__main__':
    main()


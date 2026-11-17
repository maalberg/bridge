"""
ablate_quant.py -- which tensor's fixed-point range costs the accuracy?

The tensors at the boundaries of the DPU stages are quantized to 8 bits with a power-of-two scale
(fix point f: step 2**-f, range [-128 * 2**-f, 127 * 2**-f]). A value beyond the range saturates.
This tool takes the FLOAT model and quantizes one boundary tensor at a time with the same arithmetic,
everything else staying float. What it reports:

  1. ranges      what every tensor carries, which fix point would avoid saturation, how much saturates at the
                 fix point of the deployed model (--fix)
  2. sweep       the accuracy when only one tensor is quantized, for a range of fix points: shows how sensitive
                 the model is to the range (saturation) and to the step (resolution) of each tensor
  3. combined    (with --fix) all tensors at the deployed fix points; one tensor kept in float (how much would
                 repairing it give?); one tensor with a range that avoids saturation (what if the range were wider?)

Only the quantization of the boundary tensors is emulated. The weights and the activations inside a stage are not,
so the accuracy of "all tensors at the deployed fix points" can be better than what the board measures (--reference).
If it is close, the boundary tensors explain the loss; if not, the loss is inside the stages.

The numpy glue between the stages (pipeline.py) is the one that runs on the board.
"""

import os
import json
import numpy as np

import util_plan
import util_consts
from util import load_session
from util_io import read_arguments, read_trajectories, import_module, tabulate
from util_model import numpy_stage


# --! the dpu works with 8-bit integers
NBIT_MIN, NBIT_MAX = -128, 127

# --! tensors with more values than this are subsampled for the statistics
MAX_STATS_VALUES = 2_000_000


def quantize(x, fix_point):
    """Rounds ``x`` to the grid of step 2**-fix_point and saturates it to 8 bits."""
    scale = 2.0 ** fix_point
    return (np.clip(np.round(x * scale), NBIT_MIN, NBIT_MAX) / scale).astype(np.float32)


def saturated_fraction(x, fix_point):
    """Fraction of the values of ``x`` that do not fit into the range of ``fix_point``."""
    q = np.round(x * 2.0 ** fix_point)
    return float(np.mean((q > NBIT_MAX) | (q < NBIT_MIN)))


def no_saturation_fix_point(x):
    """The largest fix point (finest step) at which no value of ``x`` saturates."""
    peak = max(float(x.max()) / NBIT_MAX, -float(x.min()) / -NBIT_MIN)
    if peak <= 0.0:
        return 7
    return int(np.floor(-np.log2(peak)))


def parse_fix(text, tensors):
    """'encoder.in=6,classifier.out=5' -> {'encoder.in': 6, 'classifier.out': 5}"""
    fix = {}
    for item in text.split(','):
        name, sep, value = item.strip().partition('=')
        if not sep or name not in tensors:
            raise SystemExit(f'[error]: --fix entries look like <stage>.in=<n> or <stage>.out=<n>; known tensors: {tensors}')
        try:
            fix[name] = int(value)
        except ValueError:
            raise SystemExit(f'[error]: --fix {name}: {value} is not an integer')
    return fix


class emulated_stage:
    """
    A float stage whose input and output tensors are optionally quantized, so that it behaves as the two
    fixed-point tensors at its boundary would make it behave. Optionally records the tensors.
    """
    def __init__(self, name, stage, in_fix=None, out_fix=None, store=None):
        self.name, self.stage, self.in_fix, self.out_fix, self.store = name, stage, in_fix, out_fix, store

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.store is not None:
            self.store.setdefault(f'{self.name}.in', []).append(x)
        if self.in_fix is not None:
            x = quantize(x, self.in_fix)

        y = np.asarray(self.stage(x), dtype=np.float32)
        if self.store is not None:
            self.store.setdefault(f'{self.name}.out', []).append(y)
        if self.out_fix is not None:
            y = quantize(y, self.out_fix)
        return y


def main():

    options = {
        '--plan': {
            'help'    : 'model plan in a .json file written by analyze_model.py',
            'default' : 'model_plan.json'
        },
        '--consts': {
            'help'    : 'model constants in a .json file written by export_constants.py',
            'default' : 'model_consts.json'
        },
        '--pipeline'  : {
            'help'    : "numpy port of model's forward method",
            'default' : 'pipeline.py',
        },
        '--labels': {
            'help'    : '.csv file with one label per raw sample, same row layout as --data (required)'
        },
        '--out-field': {
            'help'    : 'field of the pipeline output that is classified; default=score',
            'default' : 'score'
        },
        '--fix': {
            'help'    : 'fix points of the deployed model, for example encoder.in=6,encoder.out=6,classifier.in=7,classifier.out=5 '
                        '(they are printed by check_golden.py on the board); enables the combined experiments'
        },
        '--reference': {
            'type'    : float,
            'help'    : 'accuracy in percent measured on the board, printed next to the emulated one'
        },
        '--fix-min': {
            'type'    : int,
            'help'    : 'smallest fix point of the sweep; default=1',
            'default' : 1
        },
        '--fix-max': {
            'type'    : int,
            'help'    : 'largest fix point of the sweep; default=7',
            'default' : 7
        },
        '--max-traj': {
            'type'    : int,
            'help'    : 'use only the first N trajectories (faster)'
        },
        '--out': {
            'help'    : 'results are saved to this .json file; default=ablation.json',
            'default' : 'ablation.json'
        }
    }
    ns = read_arguments(description='per-tensor fixed-point ablation: which tensor range costs the accuracy?', extra_options=options)

    for path, hint in ((ns.plan, 'run analyze_model.py first'), (ns.consts, 'run export_constants.py first'),
                       (ns.pipeline, "write a numpy port of model's forward method")):
        if not os.path.isfile(path):
            raise SystemExit(f'[error]: {path} does not exist; {hint}')

    if not ns.data or not ns.traj_nsample or not ns.weights or not ns.labels:
        raise SystemExit('[error]: for the ablation, --data, --traj-nsample, --weights, and --labels are required')

    if ns.fix_min > ns.fix_max:
        raise SystemExit('[error]: --fix-min must not be larger than --fix-max')

    # --! load session, plan, constants, and the numpy pipeline, as the pipeline check does
    model, _, _, trajectories, _ = load_session(ns)
    plan, consts = util_plan.load(ns.plan), util_consts.load(ns.consts)
    port = import_module(ns.pipeline, 'pipeline', ['make_inputs', 'run', 'align_labels'])

    data = trajectories.numpy()
    labels = read_trajectories(ns.labels, ns.traj_nsample)
    if labels.shape[2] != 1:
        raise SystemExit(f'[error]: {ns.labels} must have exactly one column')
    labels = labels[:, :, 0].astype(np.int64)

    ntraj = min(ns.max_traj or len(data), len(data))

    # --! the tensors are the inputs and outputs of the stages, in the order of the pipeline
    stage_names = list(plan.get_stage_rows())
    tensors = [f'{name}.{kind}' for name in stage_names for kind in ('in', 'out')]
    float_stages = {name: numpy_stage(getattr(model, name)) for name in stage_names}

    deployed = parse_fix(ns.fix, tensors) if ns.fix else None

    def accuracy(fix, store=None):
        """Accuracy in percent of the pipeline with the tensors in ``fix`` quantized, all others float."""
        stages = {n: emulated_stage(n, float_stages[n], fix.get(f'{n}.in'), fix.get(f'{n}.out'), store) for n in stage_names}
        hits = nrow = 0
        for t in range(ntraj):
            out = port.run(port.make_inputs(data[t], consts), stages, consts)
            pred = np.argmax(np.asarray(getattr(out, ns.out_field)), axis=-1)
            hits += int((pred == port.align_labels(labels[t], consts)).sum())
            nrow += pred.shape[0]
        return 100.0 * hits / nrow

    results = {'sweep': {}, 'ranges': {}}

    # --! 0. float baseline; the same pass records the tensors
    store = {}
    float_acc = accuracy({}, store)
    values = {}
    for name, chunks in store.items():
        flat = np.concatenate([c.ravel() for c in chunks])
        if flat.size > MAX_STATS_VALUES:
            flat = flat[::int(np.ceil(flat.size / MAX_STATS_VALUES))]
        values[name] = flat
    results['float'] = float_acc

    print(f'[note]: {ntraj} trajectories, tensors {tensors}')
    print(f'[result]: float pipeline: {float_acc:.2f}%' + (f'; on the board: {ns.reference:.2f}%' if ns.reference else ''))

    # --! 1. ranges
    rows = []
    for name in tensors:
        x = values[name]
        f_ns = no_saturation_fix_point(x)
        results['ranges'][name] = {'min': float(x.min()), 'max': float(x.max()), 'std': float(x.std()),
                                   'no_saturation_fix_point': f_ns}
        row = [name, f'{x.min():.3g}', f'{x.max():.3g}', f'{x.std():.3g}', f_ns, f'{2.0 ** -f_ns:.4g}']
        if deployed is not None and name in deployed:
            f = deployed[name]
            sat = saturated_fraction(x, f)
            results['ranges'][name].update({'deployed_fix_point': f, 'deployed_saturated': sat})
            row += [f, f'[{NBIT_MIN * 2.0 ** -f:.4g}, {NBIT_MAX * 2.0 ** -f:.4g}]', f'{100.0 * sat:.2f}%']
        elif deployed is not None:
            row += ['', '', '']
        rows.append(row)

    headers = ['tensor', 'min', 'max', 'std', 'f without saturation', 'step there']
    if deployed is not None:
        headers += ['deployed f', 'deployed range', 'saturated']
    print('\n[result]: 1. what the tensors carry (float pipeline)')
    print(tabulate(headers, rows))

    # --! 2. sweep: one tensor quantized at a time
    fixes = list(range(ns.fix_min, ns.fix_max + 1))
    rows, sat_rows = [], []
    for name in tensors:
        results['sweep'][name] = {}
        row, sat_row = [name], [name]
        for f in fixes:
            acc = accuracy({name: f})
            results['sweep'][name][f] = acc
            mark = '*' if deployed is not None and deployed.get(name) == f else ''
            row.append(f'{acc:.2f}{mark}')
            sat_row.append(f'{100.0 * saturated_fraction(values[name], f):.1f}%')
        rows.append(row)
        sat_rows.append(sat_row)
        print(f'[note]: swept {name}')

    headers = ['tensor'] + [f'f={f}' for f in fixes]
    range_row = ['range +/-'] + [f'{NBIT_MAX * 2.0 ** -f:.3g}' for f in fixes]
    print(f'\n[result]: 2. accuracy in %, only this tensor quantized, all others float (float: {float_acc:.2f}%)'
          + ('; * = deployed' if deployed is not None else ''))
    print(tabulate(headers, [range_row] + rows))
    print('\n[result]:    share of the values that saturate')
    print(tabulate(headers, sat_rows))

    # --! 3. combined experiments at the deployed fix points
    if deployed is not None:
        combined = {'all deployed': accuracy(deployed)}
        for name in deployed:
            rest = {k: v for k, v in deployed.items() if k != name}
            combined[f'{name} in float'] = accuracy(rest)
            widened = dict(deployed)
            widened[name] = results['ranges'][name]['no_saturation_fix_point']
            combined[f'{name} at f={widened[name]} (no saturation)'] = accuracy(widened)
            combined[f'only {name} at f={deployed[name]}'] = accuracy({name: deployed[name]})
        results['combined'] = combined

        print('\n[result]: 3. combined, with the deployed fix points ' + str({k: deployed[k] for k in tensors if k in deployed}))
        rows = [[k, f'{v:.2f}', f'{v - combined["all deployed"]:+.2f}'] for k, v in combined.items()]
        print(tabulate(['experiment', 'accuracy %', 'against all deployed'], rows))

        if ns.reference:
            print(f'\n[result]: all deployed, emulated: {combined["all deployed"]:.2f}%; board: {ns.reference:.2f}%; '
                  f'float: {float_acc:.2f}%')
            board_loss = float_acc - ns.reference
            emulated_loss = float_acc - combined['all deployed']
            if board_loss < 0.05:
                print('[result]: the board loses nothing against float')
            elif emulated_loss > board_loss * 1.05:
                print(f'[result]: the emulation loses {emulated_loss:.2f} points, more than the {board_loss:.2f} of the board: '
                      'the boundary tensors alone are more than enough to explain the loss')
            else:
                print(f'[result]: the boundary tensors explain {100.0 * emulated_loss / board_loss:.0f}% of the {board_loss:.2f} points '
                      'that the board loses against float (the rest is quantization inside the stages)')

    with open(ns.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[note]: results saved to {ns.out}')


if __name__ == '__main__':
    main()

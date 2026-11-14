
import copy
import os

import torch

from util import load_session
from util_io import read_arguments, read_trajectories
from util_model import process_stage, torch_process_stage, get_stage_names, capture_first_rows, extract_output


def main():

    # --! read arguments with extra testing-specific arguments and options
    options = {
        '--labels': {
            'help'    : '.csv file with one label per raw sample, same row layout as --data'
        },
        '--out-field': {
            'help'    : 'field of model output to test',
            'default' : 'score'
        },
        '--check-alignment': {
            'action'  : 'store_true',
            'help'    : 'with --labels: float accuracy for predictions shifted against labels'
        }
    }
    ns = read_arguments(description='test quantized composite model end to end', extra_options=options)

    # --! we do not want to change the properties of base options, so impose extra constraints on them at run time
    if not ns.data or not ns.traj_nsample or not ns.weights:
        raise SystemExit('[error]: for testing, --data, --traj-nsample, and --weights are required')

    if ns.check_alignment and not ns.labels:
        raise SystemExit('[error]: --check-alignment needs --labels')

    # --! load current session
    model, _, weights_check, trajectories, input_sets = load_session(ns)

    # --! report on data trajectories read
    print(f'[note]: read {trajectories.shape[0]} trajectories of shape {tuple(trajectories.shape[1:])} from {ns.data}')

    # --! labels, aligned with rows of model output
    aligned = None
    if ns.labels:
        if not hasattr(model, 'align_labels'):
            raise SystemExit('[error]: --labels needs align_labels(labels) method in model')
        labels = torch.from_numpy(read_trajectories(ns.labels, ns.traj_nsample))
        if labels.shape[2] != 1:
            raise SystemExit(f'[error]: {ns.labels} must have exactly one column')
        print(f'[note]: read {labels.shape[0]} labels of shape {tuple(labels.shape[1:])} from {ns.labels}')
        aligned = model.align_labels(labels[:, :, 0].long())

    # --! create a copy of composite model for quantization
    qmodel = copy.deepcopy(model)

    # --! extract first data samples from the sets to serve as 'dummy' inputs that go into stages
    first = capture_first_rows(model, get_stage_names(model), input_sets)
    stages = [s for s in get_stage_names(model) if s in first]
    skipped = [s for s in get_stage_names(model) if s not in first]
    if skipped:
        print(f'[warn]: stages never called, left untouched: {skipped}')

    # --! every quantized stage lives in a process of its own: two quantizers in one process do not work
    workers = {name: process_stage(ns, name, os.path.join(ns.out_quant, name)) for name in stages}
    for name, worker in workers.items():
        setattr(qmodel, name, torch_process_stage(worker))
    print(f'[note]: testing quantized stages {stages} on {len(input_sets)} trajectories...')

    # --! run float and quantized composite on every trajectory
    nrow = nagree = ncorrect_f = ncorrect_q = 0
    max_diff = 0.0
    float_preds = []

    with torch.no_grad():
        for t, inputs in enumerate(input_sets):
            sf = extract_output(model(*inputs), ns.out_field)
            sq = extract_output(qmodel(*inputs), ns.out_field)
            pf, pq = sf.argmax(dim=-1), sq.argmax(dim=-1)

            nrow += pf.shape[0]
            nagree += int((pf == pq).sum())
            max_diff = max(max_diff, float((sf - sq).abs().max()))

            if aligned is not None:
                y = aligned[t]
                if y.shape[0] != pf.shape[0]:
                    raise SystemExit(
                        f'[error]: align_labels gives {y.shape[0]} labels for trajectory {t}, but the model produces {pf.shape[0]} outputs')
                ncorrect_f += int((pf == y).sum())
                ncorrect_q += int((pq == y).sum())
                float_preds.append(pf)
            print(f'[note]: trajectory {t + 1}/{len(input_sets)} done')

    print('')
    print(f'[result]: {len(input_sets)} trajectories, {nrow} classified rows')
    if aligned is not None:
        print(f'[result]: float accuracy     : {(100.0 * ncorrect_f / nrow):.1f}%')
        print(f'[result]: quantized accuracy : {(100.0 * ncorrect_q / nrow):.1f}%')
    print(f'[result]: float/quantized agreement : {(100.0 * nagree / nrow):.1f}%')
    print(f'[result]: max |score difference|    : {max_diff:.3f}')

    # --! does the label alignment look right? accuracy should peak at shift 0
    if ns.check_alignment:
        print('')
        print('[alignment]: float accuracy with predictions shifted against labels')
        for s in range(-4, 5):
            correct = count = 0
            for pf, y in zip(float_preds, aligned):
                m = pf.shape[0]
                a, l = (pf[s:], y[:m - s]) if s >= 0 else (pf[:m + s], y[-s:])
                correct += int((a == l).sum())
                count += a.shape[0]
            print(f'[alignment]: shift {s:+d}: {(100.0 * correct / count):.2f}%')

    for worker in workers.values():
        worker.close()


if __name__ == '__main__':
    main()


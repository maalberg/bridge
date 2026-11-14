
import os
import copy
import torch
import numpy as np


import util_plan
import util_consts
from util import load_session
from util_io import read_arguments, read_trajectories, import_module
from util_dpu import capture
from util_model import get_stage_names, capture_first_rows, numpy_stage, stage_recorder, process_stage, torch_process_stage


def fields(output):
    """A model output as {name: numpy array}."""
    if hasattr(output, '_fields'):
        return {f: _np(getattr(output, f)) for f in output._fields}
    return {'output': _np(output)}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def get_relative_diff(a, b):
    """(max |a - b|, relative to the largest magnitude of b)"""
    d = float(np.abs(a - b).max())
    return d, d / max(float(np.abs(b).max()), 1e-12)


def unique_in_order(items):
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


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
        '--tol': {
            'type'    : float,
            'help'    : 'relative tolerance of float comparison; default=1e-3',
            'default' : 1e-3,
        },
        '--no-quant': {
            'action'  : 'store_true',
            'help'    : 'skip check 3: quantization'
        },
        '--labels': {
            'help'    : '.csv file with one label per raw sample, same row layout as --data; adds check 5 and the accuracies'
        },
        '--golden': {
            'help'    : 'save outputs of check 3 on trajectory 0 to this .npz file'
        }
    }
    ns = read_arguments(description='verify numpy pipeline against torch model', extra_options=options)

    if not os.path.isfile(ns.plan):
        raise SystemExit(f'[error]: model plan {ns.plan} does not exist; run analyze_model.py first')

    if not os.path.isfile(ns.consts):
        raise SystemExit(f'[error]: model constants {ns.consts} do not exist; run export_constants.py first')

    if not os.path.isfile(ns.pipeline):
        raise SystemExit(f"[error]: pipeline {ns.pipeline} does not exist; write a numpy port of model's forward method")

    if ns.golden and ns.no_quant:
        raise SystemExit('[error]: --golden needs check 3, drop --no-quant')

    # --! a pipeline check on a model without its weights would pass but prove nothing
    if not ns.data or not ns.traj_nsample or not ns.weights:
        raise SystemExit('[error]: for pipeline checking, --data, --traj-nsample, and --weights are required')

    # --! np.savez appends .npz when it is missing; do the same here so that the reported path is the real one
    if ns.golden and not ns.golden.endswith('.npz'):
        ns.golden += '.npz'

    # --! load current session
    model, _, _, trajectories, input_sets = load_session(ns)

    # --! load model plan and constants
    plan, consts = util_plan.load(ns.plan), util_consts.load(ns.consts)

    # --! import pipeline module, model's forward method ported to numpy
    port = import_module(ns.pipeline, 'pipeline', ['make_inputs', 'run'])

    # --! labels are needed for check 5 and for the accuracies
    labels = None
    if ns.labels:
        for obj, what in ((model, 'model'), (port, ns.pipeline)):
            if not hasattr(obj, 'align_labels'):
                raise SystemExit(f'[error]: --labels needs align_labels in {what}')
        labels = torch.from_numpy(read_trajectories(ns.labels, ns.traj_nsample))
        if labels.shape[2] != 1:
            raise SystemExit(f'[error]: {ns.labels} must have exactly one column')
        labels = labels[:, :, 0].long()

    def count_hits(t, scores):
        """Number of correct predictions of the numpy pipeline on trajectory ``t``."""
        return int((np.argmax(scores, -1) == port.align_labels(labels[t].numpy(), consts)).sum())

    # --! check that model adhers to a single-parameter interface: model(param)
    for inputs in input_sets:
        if len(inputs) != 1:
            raise SystemExit('[error]: this check supports models with one input tensor')

    # --! create numpy wrappers for torch-based stages
    names = get_stage_names(model)
    float_stages = {n: numpy_stage(getattr(model, n)) for n in names}

    # --! number of input sets corresponds to number of trajectories
    ntraj = len(input_sets)

    # --! start checks in good - okay - state
    ok = True

    # --! checks 1 and 2: make_inputs and float pipeline
    worst = {}
    input_diff = 0.0
    float_hits = float_nrow = 0
    for t in range(ntraj):

        # --! check making of inputs between reference model and ported one
        past = port.make_inputs(trajectories[t].numpy(), consts)
        ref_past = input_sets[t][0].numpy()
        if past.shape != ref_past.shape:
            raise SystemExit(f'[error]: make_inputs gives shape {past.shape}, model gives {ref_past.shape}')
        input_diff = max(input_diff, float(np.abs(past - ref_past).max()))

        # --! check output field by field between reference model and ported one
        with torch.no_grad():
            ref_out = fields(model(*input_sets[t]))
        out = fields(port.run(past, float_stages, consts))
        for f in ref_out:
            if f in out:
                worst[f] = max(worst.get(f, 0.0), get_relative_diff(out[f], ref_out[f])[1])
        if labels is not None:
            float_hits += count_hits(t, out['score'])
            float_nrow += out['score'].shape[0]

    res = 'PASS' if input_diff == 0.0 else 'FAIL'
    print(f'[check 1]: make_inputs, max |difference| to torch windows: {input_diff:.3g} -> {res}')
    ok = ok and input_diff == 0.0
    for f, r in worst.items():
        passed = r <= ns.tol
        ok = ok and passed
        print('[check 2]: float pipeline vs model.forward, %-6s relative difference %.3g  -> %s'
              % (f, r, 'PASS' if passed else 'FAIL'))

    # --! check 5: label alignment of numpy port against model
    if labels is not None:
        ref_labels = model.align_labels(labels)
        same = all(np.array_equal(port.align_labels(labels[t].numpy(), consts), _np(ref_labels[t])) for t in range(ntraj))
        ok = ok and same
        print('[check 5]: align_labels, numpy port vs model  -> %s' % ('PASS' if same else 'FAIL'))
        print('[accuracy]: float numpy pipeline, %d rows: %.2f%%' % (float_nrow, 100.0 * float_hits / float_nrow))

    # --! check 4: stage calls and constants against plan
    log = []
    recorded = {n: stage_recorder(n, float_stages[n], log) for n in names}
    past0 = port.make_inputs(trajectories[0].numpy(), consts)
    consts.reads.clear()                      # only what run() reads is compared with forward
    port.run(past0, recorded, consts)

    plan_order = unique_in_order([e['name'] for e in plan.pipeline if e['kind'] == 'stage'])
    seen_order = unique_in_order(log)
    order_ok = plan_order == seen_order
    print('[check 4]: stage order   forward %s, port %s  -> %s'
          % (plan_order, seen_order, 'PASS' if order_ok else 'FAIL'))

    calls_ok = True
    if 'trace' in plan and 'trace_ntraj' in plan:
        for n in seen_order:
            expected = plan.trace[n]['ncalls'] / float(plan.trace_ntraj)
            if log.count(n) != expected:
                calls_ok = False
                print('[check 4]: stage %s is called %d times per trajectory, forward %g times'
                      % (n, log.count(n), expected))
    print('[check 4]: stage calls per trajectory  -> %s' % ('PASS' if calls_ok else 'FAIL'))

    used = set()
    for e in plan.pipeline + [e for calls in plan.helpers.values() for e in calls]:
        used.update(u[len('self.'):] for u in e['uses'] if u.startswith('self.'))
    missing, extra = used - consts.reads, consts.reads - used
    consts_ok = not missing
    print('[check 4]: constants     forward uses %s' % sorted(used))
    if missing:
        print('[check 4]:   not read by the port: %s' % sorted(missing))
    if extra:
        print('[check 4]:   read by the port but not used by forward: %s' % sorted(extra))
    print('[check 4]: constants  -> %s' % ('PASS' if consts_ok else 'FAIL'))
    ok = ok and order_ok and calls_ok and consts_ok

    # --! check 3: quantized path
    if not ns.no_quant:
        qmodel = copy.deepcopy(model)
        first = capture_first_rows(model, names, [input_sets[0]])

        # --! every quantized stage lives in a process of its own: two quantizers in one process do not work,
        # --! the first one silently returns float outputs
        workers = {n: process_stage(ns, n, os.path.join(ns.out_quant, n)) for n in names if n in first}
        for n, worker in workers.items():
            setattr(qmodel, n, torch_process_stage(worker))
        quant_stages = {n: workers[n] if n in workers else numpy_stage(getattr(qmodel, n)) for n in names}

        nagree = nrow = quant_hits = 0
        max_diff = 0.0
        golden = None
        for t in range(ntraj):
            past = port.make_inputs(trajectories[t].numpy(), consts)
            with torch.no_grad():
                ref = fields(qmodel(*input_sets[t]))

            # --! on trajectory 0, record what flows through the pipeline (raw data, windows, input and output
            # --! of every stage), so that the board can tell where it departs from the host
            stages_t = quant_stages
            if t == 0:
                store = {'raw_trajectory': trajectories[t].numpy(), 'raw_windows': past}
                stages_t = {n: capture(n, s, store) for n, s in quant_stages.items()}

            got = fields(port.run(past, stages_t, consts))
            key = 'score' if 'score' in ref else list(ref)[-1]
            nagree += int((np.argmax(got[key], -1) == np.argmax(ref[key], -1)).sum())
            nrow += ref[key].shape[0]
            max_diff = max(max_diff, float(np.abs(got[key] - ref[key]).max()))
            if labels is not None:
                quant_hits += count_hits(t, got[key])
            if t == 0:
                golden = dict(got)
                golden.update(store)

        agree = nagree / float(nrow)
        passed = agree >= 0.999
        ok = ok and passed
        print('[check 3]: quantized pipeline vs quantized torch composite, %s: argmax agreement %.3f%%, '
              'max |difference| %.4g  -> %s' % (key, 100.0 * agree, max_diff, 'PASS' if passed else 'FAIL'))
        if labels is not None:
            print('[accuracy]: quantized numpy pipeline, %d rows: %.2f%%   <- the board should reproduce this'
                  % (nrow, 100.0 * quant_hits / nrow))
        if ns.golden:
            np.savez(ns.golden, **golden)
            print(f'[note]: saved quantized pipeline output of trajectory 0 to {ns.golden}')

    print('')
    for worker in locals().get('workers', {}).values():
        worker.close()
    print('[result]: %s' % ('all checks passed' if ok else 'SOME CHECKS FAILED'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

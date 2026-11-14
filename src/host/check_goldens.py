"""
check_goldens.py -- the golden vectors of the pipeline and of the single stages must agree.

    python3 check_goldens.py golden_pipeline.npz golden_encoder.npz golden_classifier.npz

golden_pipeline.npz records, on trajectory 0, the rows every stage received in the quantized pipeline and what it
returned. golden_<stage>.npz records the same stage run alone. Where both saw the same input rows, the outputs must be
identical. If they are not, a stage was quantized in one run but not in the other, and one of the two goldens (and
every accuracy computed with it) does not describe the model that is deployed.

A stage that receives the output of an earlier quantized stage sees other rows than in its own golden, which is built
from the float pipeline; it is not comparable and is reported as such.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description='compare the golden vectors of the pipeline with those of the stages')
    ap.add_argument('pipeline', help='golden_pipeline.npz')
    ap.add_argument('stages', nargs='+', help='golden_<stage>.npz files')
    ns = ap.parse_args()

    pipeline = np.load(ns.pipeline)
    ok, ncompared = True, 0

    for path in ns.stages:
        name = os.path.basename(path)[len('golden_'):-len('.npz')]
        stage = np.load(path)

        if f'in_{name}' not in pipeline.files:
            print(f'[warn]: {ns.pipeline} has no record of stage {name}; skipped')
            continue

        x_pipe, y_pipe = pipeline[f'in_{name}'], pipeline[f'out_{name}']
        x_stage, y_quant, y_float = stage['x'], stage['y_quant'], stage['y_float']

        if x_pipe.shape != x_stage.shape or np.abs(x_pipe - x_stage).max() > 1e-6:
            print(f'[check]: stage {name}: input rows differ between the goldens (upstream quantization); not comparable')
            continue

        ncompared += 1
        diff = float(np.abs(y_pipe - y_quant).max())
        passed = diff == 0.0
        ok = ok and passed
        print(f'[check]: stage {name}: same input rows, output in the pipeline vs alone: '
              f'max |difference| {diff:.4g}  -> {"PASS" if passed else "FAIL"}')
        if not passed:
            to_float = float(np.abs(y_pipe - y_float).max())
            print(f'[check]:   max |pipeline output - float output| is {to_float:.4g}'
                  + ('; the pipeline ran this stage as the FLOAT model' if to_float < 1e-4 else ''))

    if ncompared == 0:
        print('[check]: no stage with identical input rows: nothing could be compared')
        ok = False

    print('[result]: %s' % ('goldens agree' if ok else 'GOLDENS DISAGREE'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

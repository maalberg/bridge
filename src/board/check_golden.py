
import argparse
import time
import numpy as np

import util_dpu


def main():
    ap = argparse.ArgumentParser(description='Compare a DPU stage with golden vectors.')
    ap.add_argument('xmodel', help='compiled xmodel of the stage')
    ap.add_argument('golden', help='golden_<stage>.npz from dump_golden.py')
    ap.add_argument('--tol-lsb', type=float, default=1.0,
                    help='tolerated difference in LSB (default: 1)')
    ap.add_argument('--min-fraction', type=float, default=0.99,
                    help='fraction of outputs that must be within the tolerance (default: 0.99)')
    ap.add_argument('--argmax', action='store_true',
                    help='also report agreement of the argmax over the last axis (classifiers)')
    ns = ap.parse_args()

    g = np.load(ns.golden)
    x, y_float, y_quant = g['x'], g['y_float'], g['y_quant']

    stage = util_dpu.stage(ns.xmodel, in_shape=tuple(x.shape[1:]), out_shape=tuple(y_quant.shape[1:]))
    print(stage.describe())

    t0 = time.perf_counter()
    y_dpu = stage(x)
    seconds = time.perf_counter() - t0

    lsb = stage.out_scale
    d = (y_dpu - y_quant) / lsb                       # difference in LSB
    ad = np.abs(d)

    print('')
    print('[result]: %d rows, output rows of shape %s, one LSB = %.6g' % (x.shape[0], y_dpu.shape[1:], lsb))
    print('[result]: DPU vs fake-quantized torch (difference in LSB)')
    print('[result]:   identical        : %6.2f%%' % (100.0 * np.mean(ad < 1e-6)))
    print('[result]:   within 1 LSB     : %6.2f%%' % (100.0 * np.mean(ad <= 1.0 + 1e-6)))
    print('[result]:   within 2 LSB     : %6.2f%%' % (100.0 * np.mean(ad <= 2.0 + 1e-6)))
    print('[result]:   mean / max       : %.3f / %.3f LSB' % (ad.mean(), ad.max()))

    print('[result]: mean |error| against the float stage')
    print('[result]:   DPU              : %.5f' % np.abs(y_dpu - y_float).mean())
    print('[result]:   fake-quantized   : %.5f' % np.abs(y_quant - y_float).mean())

    lo, hi = y_dpu.min(), y_dpu.max()
    print('[result]: DPU output range [%.4g, %.4g], %.2f%% of outputs at the limits'
          % (lo, hi, 100.0 * np.mean((y_dpu <= -128 * lsb + 1e-9) | (y_dpu >= 127 * lsb - 1e-9))))

    if ns.argmax:
        agree = np.mean(np.argmax(y_dpu, axis=-1) == np.argmax(y_quant, axis=-1))
        print('[result]: argmax agreement : %.2f%%' % (100.0 * agree))

    print('[result]: %.3f ms per row (quantize, DPU call, dequantize)' % (1000.0 * seconds / x.shape[0]))

    ok = np.mean(ad <= ns.tol_lsb + 1e-6) >= ns.min_fraction
    print('[result]: %s (%.0f%% of outputs within %g LSB required)'
          % ('PASS' if ok else 'FAIL', 100.0 * ns.min_fraction, ns.tol_lsb))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

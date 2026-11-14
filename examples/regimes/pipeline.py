"""
pipeline.py -- numpy port of the CPU part of the regimes model (model.forward).

This is the model-specific glue between the DPU stages. Only numpy is used, so
the same code runs on the host and on the board. The DPU stages are passed in
as callables (float32 rows in, float32 rows out): util_dpu.stage on the board, torch
modules on the host.

Contract used by bridge_run.py and check_pipeline.py:

    make_inputs(trajectory, c)   raw trajectory [T, C] -> input of run()
    run(past, stages, c)         stages: {name: callable}, c: constants
    align_labels(labels, c)      labels [T] -> labels aligned with the rows of the score

The comments name the lines of model.py (as listed in model_plan.json) that
each statement ports.
"""

from collections import namedtuple

import numpy as np


output = namedtuple('output', ['embed', 'sig', 'score'])


def make_inputs(trajectory, constants):
    """
    Port of model.make_inputs for ONE trajectory: all windows of past_nsample
    samples, one per time step. [T, C] -> [N, past_nsample, C], N = T - past_nsample + 1.
    """
    n = constants.past_nsample
    return np.stack([trajectory[s - n:s] for s in range(n, trajectory.shape[0] + 1)])


def make_windows(traj, window_nsample, window_fun=lambda w: w):
    """Port of model.make_windows (lines 113-117): sliding windows of 2*window_nsample samples."""
    nsample = traj.shape[0]
    return np.stack([window_fun(traj[center - window_nsample:center + window_nsample])
                     for center in range(window_nsample, nsample - window_nsample)])


def run(past, stages, c):
    """
    Port of model.forward for ONE trajectory.
    past: [N, past_nsample, C] raw windows -> output(embed, sig, score)
    """
    # --! normalize input data (line 86)
    past = (past - c.in_mean) / c.in_std

    # --! encode past window into latent representation (line 89);
    # --! torch.flatten(past, start_dim=-2) merges the last two axes
    z = stages['encoder'](past.reshape(past.shape[:-2] + (-1,)))

    # --! latent representation -> signatures: eigenvalues of the covariance of a
    # --! sliding window (lines 92-95); torch.cov(w.T) is np.cov(w.T), both unbiased
    sig = make_windows(z, c.cov_nsample, lambda w: np.linalg.eigvalsh(np.cov(w.T)))

    # --! normalize signatures (line 98); float32 again, as in torch
    sig = ((sig - c.sig_mean) / c.sig_std).astype(np.float32)

    # --! classify (lines 101-103); torch.transpose(x, -1, -2) is swapaxes
    score = stages['classifier'](np.swapaxes(make_windows(sig, c.cls_nsample), -1, -2))

    return output(embed=z, sig=sig, score=score)


def align_labels(labels, c):
    """
    One label per raw sample [T] -> one label per row of the score. A window's
    label is the sample right after its past window; see model.align_labels.
    """
    first = c.past_nsample + c.cov_nsample + c.cls_nsample
    m = len(labels) - c.past_nsample + 1 - 2 * (c.cov_nsample + c.cls_nsample)
    return labels[first:first + m]

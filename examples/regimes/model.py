import torch
from collections import namedtuple

class encoder(torch.nn.Module):
    """ Encodes future-consistent latent representation for subsequent regime classification. """

    def __init__(self, window_nsample, window_ndim, latent_ndim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(window_nsample * window_ndim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_ndim)
        )

    def forward(self, x):
        return self.net(x)


class classifier(torch.nn.Module):
    """ Classifies latent dynamical signatures to identify nominal and excursion regimes. """

    def __init__(self, input_ndim=3):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=input_ndim,
                out_channels=16,
                kernel_size=5,
                padding=2
            ),
            torch.nn.ReLU(),

            torch.nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Linear(32, 2)

    def forward(self, x):

        x = self.features(x)
        x = torch.mean(x, dim=-1)
        return self.classifier(x)


model_output = namedtuple('model_output', ['embed', 'sig', 'score'])


class model(torch.nn.Module):

    def __init__(self, past_nsample, cov_nsample, cls_nsample, window_ndim, embed_ndim):
        super().__init__()

        self.encoder = encoder(past_nsample, window_ndim, embed_ndim)
        self.classifier = classifier(embed_ndim)

        self.past_nsample = past_nsample
        self.cov_nsample = cov_nsample
        self.cls_nsample = cls_nsample

        # --! standard normalization constants of signatures
        #
        # --! we register constants as buffers so that they are saved with model state
        self.register_buffer('sig_mean', torch.zeros(embed_ndim, dtype=torch.float32))
        self.register_buffer('sig_std', torch.ones(embed_ndim, dtype=torch.float32))

        # --! standard normalization constants of input data
        #
        # --! we register constants as buffers so that they are saved with model state
        self.register_buffer('in_mean', torch.zeros(window_ndim, dtype=torch.float32))
        self.register_buffer('in_std', torch.ones(window_ndim, dtype=torch.float32))

    def forward(self, past):
        """
        Runs model's forward pass.

        Required parameters:
            past    3d torch tensor with ``past`` windows of a trajectory shaped as [npast, past_nsample, nchan]

        Returns:
            score   2d torch tensor with classifier scores shaped as [nscore, 2]
        """

        # --! normalize input data
        past = (past - self.in_mean) / self.in_std

        # --! encode past window into latent representation
        z = self.encoder(torch.flatten(past, start_dim=-2))

        # --! convert latent representation into normalized signatures
        sig = self.make_windows(
            z,
            window_nsample=self.cov_nsample,
            window_fun=lambda w: torch.linalg.eigvalsh(torch.cov(w.T)))

        # --! normalize signatures
        sig = (sig - self.sig_mean) / self.sig_std

        # --! classify signatures based on their dynamical behavior
        score = self.classifier(
            torch.transpose(
                self.make_windows(sig, window_nsample=self.cls_nsample), -1, -2))

        return model_output(embed=z, sig=sig, score=score)

    def make_windows(self, traj, window_nsample=16, window_fun=lambda w: w):
        windows = []
        nsample = traj.shape[0]

        for center in range(window_nsample, nsample - window_nsample):
            window = traj[center-window_nsample:center+window_nsample]
            windows.append(window_fun(window))

        return torch.stack(windows)

    def make_inputs(self, trajectories):
        """
        Makes model inputs from given ``trajectories``.
        The ``trajectories`` are split into past windows of size past_nsample in a sliding window fashion.

        Required parameters:
            trajectories    3d torch tensor with shape [ntraj, traj_nsample, nchan]

        Returns:
            inputs          4d torch tensor with shape [ntraj, npast, past_nsample, nchan]
        """
        return trajectories.unfold(1, self.past_nsample, 1).transpose(-1, -2)

    def align_labels(self, labels):
        """
        Aligns ``labels`` with classifier scores.
        The number of ``labels`` is expected to match the number of trajectory samples.
        The number of aligned labels will match the number of rows in the scores.
        This method omits future window length when calculating the position of the last labeled sample in the trajectories.

        Required parameters:
            labels          2d torch tensor shaped as [ntraj, traj_nsample]

        Returns:
            aligned_labels  2d torch tensor shared as [ntraj, nscore]
        """

        w, r = self.cov_nsample, self.cls_nsample
        first = self.past_nsample + w + r
        last = labels.shape[1] - w - r + 1

        return labels[:, first:last]
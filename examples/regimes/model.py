import torch


class classifier_dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels, window_nsample=16):

        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels)
        self.window_nsample = window_nsample

        self.index = []
        ntraj = self.data.shape[0]
        nsample = self.data.shape[1]

        for traj_id in range(ntraj):
            for center in range(window_nsample, nsample - window_nsample):

                self.index.append(
                    (traj_id, center, self.labels[traj_id, center])
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        traj_id, center, label = self.index[idx]
        window = self.data[traj_id, center-self.window_nsample:center+self.window_nsample]

        return (
            window,
            label
        )


class classifier(torch.nn.Module):

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


    def test_output(self, output, label):
        o_exc = (output[:, 1] - output[:, 0]) > 0.0
        o_exc = int(o_exc)
        y_exc = int(label)
        return o_exc == y_exc



import torch
import numpy as np


def normalize_standard(timeseries, mean, std):
    return (timeseries - mean) / std


def normalize_calib(data, std_min=1e-3):

    std_min = torch.tensor(std_min, dtype=torch.float32)

    data_mean = [d.mean() for d in torch.split(data, 1, dim=-1)]
    data_std = [torch.maximum(d.std(), std_min) for d in torch.split(data, 1, dim=-1)]
    data = torch.cat([
        normalize_standard(
            d, mean, std) for d, mean, std in zip(
                torch.split(data, 1, dim=-1),
                data_mean,
                data_std)], dim=-1)

    # --! convert lists with data mean and standard deviation into row vectors
    data_mean = torch.as_tensor(data_mean, dtype=torch.float32).reshape(1, -1)
    data_std = torch.as_tensor(data_std, dtype=torch.float32).reshape(1, -1)

    return data, data_mean, data_std


def normalize_test(data, data_mean, data_std):

    data = torch.cat([
        normalize_standard(
            d, mean, std) for d, mean, std in zip(
                torch.split(data, 1, dim=-1),
                data_mean,
                data_std)], dim=-1)

    return data


def read_file(name: str, datachunk_len, delim=',') -> torch.Tensor:
    """
    Reads data from a file called ``name`` and formats the data based on ``datachunk_len``,
    i.e. the length of one contiguous chunk of data. The file data are expected to be
    in format [T, C], such that the read data could be formatted as [B, T, C],
    where B, T and C are the number of batches, time steps and data channels,
    respectively.
    """

    # --! note that we force numpy loadtxt to return at least a two-dimensional array
    # --! by setting ndmin=2
    data = torch.tensor(
        np.loadtxt(name + '.csv', delimiter=delim, dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))
    

def write_file(name: str, data, delimiter: str = ','):
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt(name + '.csv', filedata, fmt='%.14f', delimiter=delimiter)



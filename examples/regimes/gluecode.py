import sys
import numpy as np
import vart
import xir


def read_datafile(name: str, datachunk_len: int, delim: str = ',') -> np.ndarray:
    data = np.loadtxt(name + '.csv', delimiter=delim, dtype=np.float32, ndmin=2)
    datachunks_n = int(data.shape[0] / datachunk_len)
    return data.reshape(datachunks_n, datachunk_len, data.shape[1])


def normalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std


def make_windows(data: np.ndarray, labels: np.ndarray, window_nsample: int):

    """
    data:   [ntraj, nsample, C]
    labels: [ntraj, nsample]
    Returns:
      windows:       [N, 2*window_nsample, C]
      window_labels: [N]
      index:         list of (traj_id, center) for traceability
    """

    ntraj, nsample, c = data.shape
    windows = []
    window_labels = []
    index = []
    for traj_id in range(ntraj):
        for center in range(window_nsample, nsample - window_nsample):
            windows.append(data[traj_id, center - window_nsample:center + window_nsample])
            window_labels.append(labels[traj_id, center])
            index.append((traj_id, center))
    return np.stack(windows, axis=0), np.array(window_labels), index


def get_dpu_subgraphs(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()

    return [
        subgraph for subgraph in child_subgraphs
        if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU"]


def test_o(o, label):
    o_exc = (o[:, 1] - o[:, 0]) > 0.0
    o_exc = int(o_exc)
    y_exc = int(label)
    return int(o_exc == y_exc)


def main(argv):

    # --! there are 2 normalization constants that we read from a file: mean and standard deviation
    norm_filename = 'norm_const'
    norm_nconst = 2
    norm_consts = read_datafile(norm_filename, norm_nconst)
    print(f"[note]: read normalization constants with shape {norm_consts.shape} from file {norm_filename}")

    data_filename = 'data'
    data_nsample = 1152
    data = read_datafile(data_filename, data_nsample)
    print(f"[note]: read data with shape {data.shape} from file {data_filename}")
    data = normalize_data(data, norm_consts[0, 0], norm_consts[0, 1])

    labels_filename = 'labels'
    labels_nsample = 1152
    labels = read_datafile(labels_filename, labels_nsample)
    print(f"[note]: read labels with shape {labels.shape} from file {labels_filename}")

    window_nsample = 16
    windows, labels, indeces = make_windows(data, labels, window_nsample)
    #windows = np.transpose(windows, (0, 2, 1))
    labels = np.squeeze(labels, -1)
    print(f"[note]: extracted windows with shape {windows.shape} and labels with shape {labels.shape}")

    graph = xir.Graph.deserialize(argv[1])
    dpu_subgraphs = get_dpu_subgraphs(graph)
    print(f"[note]: there are {len(dpu_subgraphs)} dpu subgraphs in model {argv[1]}")

    dpu_subgraph = dpu_subgraphs[0]
    dpu_runner = vart.Runner.create_runner(dpu_subgraph, 'run')

    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()
    for i in input_tensors:
        print(f'[note]: input tensor {i.name} has shape {i.dims} and type {i.dtype}')
    for o in output_tensors:
        print(f'[note]: output tensor {o.name} has shape {o.dims} and type {o.dtype}')

    i_ndim = tuple(input_tensors[0].dims)
    o_ndim = tuple(output_tensors[0].dims)

    i_fixpos = input_tensors[0].get_attr("fix_point")
    i_scale = 2 ** i_fixpos
    o_fixpos = output_tensors[0].get_attr("fix_point")
    o_scale = 1.0 / (2 ** o_fixpos)

    ncorrect = 0
    for w, l in zip(windows, labels):
        quant_i = np.clip(np.round(w * i_scale), -128, 127).astype(np.int8)

        i_data = [np.empty(i_ndim, dtype=np.int8, order="C")]
        i_data[0][0] = quant_i

        o_data = [np.empty(o_ndim, dtype=np.int8, order="C")]

        job_id = dpu_runner.execute_async(i_data, o_data)
        dpu_runner.wait(job_id)

        o = o_data[0].astype(np.float32) * o_scale
        ncorrect = ncorrect + test_o(o, l)

    acc = ncorrect / len(windows) * 100
    print(f"[note]: on-target accuracy over {len(windows)} windows: {acc:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage : python3 {sys.argv[0]} <xmodel_file>")
    else:
        main(sys.argv)

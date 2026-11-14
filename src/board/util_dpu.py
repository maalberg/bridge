import numpy as np
import itertools

from util_io import tabulate


# --! element types of dpu tensors, as reported by xir
dtypes = {
    'xint8'   : np.int8,
    'xuint8'  : np.uint8,
    'xint16'  : np.int16,
    'xuint16' : np.uint16,
}


# --! usual number of non-dpu subgraphs in a typical graph;
# --! the two usual subgraphs (likely) serve as data converters from float to fixed representation and back
nondpu_nsubgraph_usual = 2


def find_subgraphs(graph):
    """
    Finds dpu subgraphs in a given ``graph``.

    Required parameters:
        graph       graph, graph object deserialized using ``xir.Graph.deserialize`` method

    Returns:
        dpu         list, dpu subgraphs
        other       list, non-dpu subgraphs found in given ``graph``
    """
    root = graph.get_root_subgraph()
    if root.is_leaf:
        return [], []
    dpu, other = [], []
    for subgraph in root.toposort_child_subgraph():
        if subgraph.has_attr('device') and subgraph.get_attr('device').upper() == 'DPU':
            dpu.append(subgraph)
        else:
            other.append(subgraph)
    return dpu, other


def get_op_types(subgraph):
    """
    Gets operator types in a ``subgraph``.

    Required parameters:
        subgraph        graph, a subgraph with operators

    Returns:
        op_types        list of strings or None, a sorted list of operator names
    """
    try:
        return sorted(str(op.get_type()) for op in subgraph.get_ops())
    except Exception:
        return None


def get_representable_range(tensor):
    """
    Gets a range of float values this int ``tensor`` with its fix point can hold.
    """
    info = np.iinfo(dtypes[tensor.dtype])
    scale = 2.0 ** tensor.get_attr('fix_point')
    return info.min / scale, info.max / scale


def _perm(src, dst):
    """
    Axes permutation p with np.transpose(a, p).shape == dst for any a of shape src.
    Identity if the shapes are equal; error if there is no such permutation or
    more than one (give the permutation explicitly in that case).
    """
    src, dst = tuple(src), tuple(dst)
    if src == dst:
        return tuple(range(len(src)))
    candidates = [p for p in itertools.permutations(range(len(src)))
                  if tuple(src[i] for i in p) == dst]
    if not candidates:
        raise ValueError('shape %s cannot be permuted into %s' % (src, dst))
    if len(candidates) > 1:
        raise ValueError('shape %s -> %s is ambiguous (%d permutations); pass the '
                         'permutation explicitly' % (src, dst, len(candidates)))
    return candidates[0]


def _permuted_shape(shape, perm):
    """shape of np.transpose(a, perm) for a of the given shape."""
    return tuple(shape[i] for i in perm)


def _inverse_shape(dst, perm):
    """the shape `a` must have so that np.transpose(a, perm) has shape dst."""
    src = [None] * len(perm)
    for i, p in enumerate(perm):
        src[p] = dst[i]
    return tuple(src)


def _batch_perm(perm):
    """permutation of a [rows, ...] array that permutes only the row axes."""
    return (0,) + tuple(p + 1 for p in perm)


class stage:
    def __init__(self, model, in_shape=None, out_shape=None, in_perm=None, out_perm=None):
        """
        Constructs a dpu stage from ``model`` description.
        """
        import vart
        import xir

        # --! save model instance
        self.model = model

        # --! the graph must live as long as the runner: subgraphs and runner point into it, so a graph
        # --! that is freed when this method returns leaves them with dangling pointers. with one stage
        # --! this goes unnoticed; with several, a later graph can take the freed memory of an earlier one
        self.graph = xir.Graph.deserialize(model)

        # --! extract dpu subgraphs
        dpu, other = find_subgraphs(self.graph)
        if len(dpu) != 1:
            raise ValueError(f'[error]: {model} has {len(dpu)} dpu subgraphs, but a stage needs exactly one')
        if len(other) > nondpu_nsubgraph_usual:
            print(f'[warn]: {model} contains {len(other)} non-dpu subgraph(s) ({nondpu_nsubgraph_usual} are usual)')
        self.other = other

        # --! check model inputs and outputs
        self.runner = vart.Runner.create_runner(dpu[0], 'run')
        inputs = self.runner.get_input_tensors()
        outputs = self.runner.get_output_tensors()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f'[error]: a stage needs one input and one output tensor, found {len(inputs)} and {len(outputs)}')

        # --! save data inputs and outputs for reporting purposes
        in_data, out_data = inputs[0], outputs[0]
        self.in_data, self.out_data = in_data, out_data

        # --! check data types in input, output tensors
        for data in (in_data, out_data):
            if data.dtype not in dtypes:
                raise ValueError(f'[error]: unsupported data type {data.dtype}')

        # --! save input, output data types for later use
        self.in_type = dtypes[in_data.dtype]
        self.out_type = dtypes[out_data.dtype]

        # --! get machine limits for input integer types
        info = np.iinfo(self.in_type)
        self.in_lo, self.in_hi = info.min, info.max

        # --! dpu accepts data in one-element batches, so save samples of in/out data rows
        self.batch_nelem = int(in_data.dims[0])
        self.in_row = tuple(int(data) for data in in_data.dims[1:])
        self.out_row = tuple(int(data) for data in out_data.dims[1:])

        # --! quantization scales are powers of two, given by fix point
        self.in_scale = 2.0 ** in_data.get_attr('fix_point')
        self.out_scale = 1.0 / 2.0 ** out_data.get_attr('fix_point')

        # --! layout: framework row -> dpu row (input), dpu row -> framework row (output)
        if in_perm is None and in_shape is not None:
            in_perm = _perm(in_shape, self.in_row)
        if out_perm is None and out_shape is not None:
            out_perm = _perm(self.out_row, out_shape)
        self.in_perm = tuple(in_perm) if in_perm is not None else None
        self.out_perm = tuple(out_perm) if out_perm is not None else None

        # --! check given in/out shapes
        if in_shape is not None:
            self.in_shape = tuple(in_shape)
        elif self.in_perm is not None:
            self.in_shape = _inverse_shape(self.in_row, self.in_perm)
        else:
            self.in_shape = self.in_row
        self.out_shape = (_permuted_shape(self.out_row, self.out_perm) if self.out_perm is not None else self.out_row)

        # --! buffers are reused: every call waits for its job before the next one
        self.in_buf = [np.zeros(tuple(in_data.dims), dtype=self.in_type, order='C')]
        self.out_buf = [np.zeros(tuple(out_data.dims), dtype=self.out_type, order='C')]

    def __call__(self, data):
        data = np.asarray(data, dtype=np.float32)

        if data.ndim != len(self.in_shape) + 1 or tuple(data.shape[1:]) != self.in_shape:
            raise ValueError(f'[error]: {self.model} expects rows of shape {self.in_shape}, but got input of shape {data.shape}')

        if self.in_perm is not None:
            data = np.transpose(data, _batch_perm(self.in_perm))

        nrow = data.shape[0]
        y = np.empty((nrow,) + self.out_row, dtype=np.float32)
        for start in range(0, nrow, self.batch_nelem):
            chunk = data[start:start + self.batch_nelem]
            m = chunk.shape[0]

            # --! quantize (round to nearest, saturate); a short last chunk is zero padded
            self.in_buf[0][:m] = np.clip(np.round(chunk * self.in_scale), self.in_lo, self.in_hi).astype(self.in_type)
            if m < self.batch_nelem:
                self.in_buf[0][m:] = 0

            job_id = self.runner.execute_async(self.in_buf, self.out_buf)
            self.runner.wait(job_id)

            y[start:start + m] = self.out_buf[0][:m].astype(np.float32) * self.out_scale

        if self.out_perm is not None:
            y = np.transpose(y, _batch_perm(self.out_perm))

        return y

    def describe(self):
        """
        Describes current dpu stage from board's perspective.

        Returns:
            string, description of this dpu stage
        """
        lines = [f'\n{self.model}']
        rows = []
        for kind, tensor in (('input', self.in_data), ('output', self.out_data)):
            lo, hi = get_representable_range(tensor)
            rows.append([
                f'{kind}',
                f'{tensor.name}',
                f'{list(tensor.dims)}',
                f'{tensor.dtype}',
                f'{tensor.get_attr("fix_point")}',
                f'[{lo:.4g}, {hi:.4g}]'])
        lines.append(tabulate(
            ['kind', 'name', 'dims', 'type', 'fix point', 'representable range'],
            rows))

        lines.append(
            f'\nrows: batch {self.batch_nelem}, framework in {self.in_shape} -> DPU {self.in_row}, DPU out {self.out_row} -> framework {self.out_shape}')

        lines.append(f'\nnon-dpu subgraphs: {len(self.other)}')
        rows = []
        for i, subgraph in enumerate(self.other):
            device = subgraph.get_attr('device') if subgraph.has_attr('device') else '?'
            ops = get_op_types(subgraph)
            ops = ops if ops is not None else 'n/a'
            rows.append([f'{i}', f'{device}', f'{ops}'])
        lines.append(tabulate(
            ['id', 'device', 'operators'],
            rows))

        return '\n'.join(lines)


class capture:
    """
    Wraps a stage and records the input and the output of its first call in ``store``
    as ``in_<name>`` and ``out_<name>``. Used to compare what flows through the stages
    on the host and on the board.
    """
    def __init__(self, name, stage, store):
        self.name, self.stage, self.store = name, stage, store

    def __call__(self, data):
        out = self.stage(data)
        if f'in_{self.name}' not in self.store:
            self.store[f'in_{self.name}'] = np.array(data, dtype=np.float32)
            self.store[f'out_{self.name}'] = np.array(out, dtype=np.float32)
        return out

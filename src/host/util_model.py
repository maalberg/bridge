import copy
import inspect
import importlib.util
import multiprocessing

import numpy as np
import torch


class batch_splitter(torch.nn.Module):
    """
    Splits input data batch into batch-1 data samples.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return torch.cat([self.model(data[i:i + 1]) for i in range(data.shape[0])], dim=0)


class numpy_stage:
    """
    Defines a numpy wrapper for a torch-based model stage.
    """

    def __init__(self, stage):
        self.stage = stage

    def __call__(self, x):
        with torch.no_grad():
            y = self.stage(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        return y.detach().cpu().numpy()


def _serve_stage(conn, args, stage_name, out_dir):
    """
    Body of a worker process: serves the calls of ONE quantized stage.

    The Vitis AI quantizer keeps global state. A second quantizer in the same process silently turns the
    first one into the float model: it still runs, but returns the float outputs. A quantized stage therefore
    always lives in a process of its own (the export of several stages from one process mixed them up too).
    """
    try:
        # --! imported here: the worker starts from scratch ('spawn') and must not inherit any quantizer state
        from pytorch_nndct.apis import torch_quantizer
        from util import load_session

        model = load_session(args)[0]
        stage = getattr(model, stage_name)
        quant_model = None
        conn.send(('ready', None))

        while True:
            data = conn.recv()
            if data is None:
                break
            x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

            # --! the quantizer needs a sample to trace the stage with: the first call provides it
            if quant_model is None:
                quantizer = torch_quantizer(
                    'test', stage, (x[:1].clone(),), output_dir=out_dir, device=torch.device('cpu'))
                quant_model = batch_splitter(quantizer.quant_model.eval())

            with torch.no_grad():
                conn.send(('ok', quant_model(x).detach().cpu().numpy()))
    except Exception as e:
        conn.send(('error', f'{type(e).__name__}: {e}'))


class process_stage:
    """
    A quantized stage that lives in a process of its own, called like any numpy stage: float32 rows in, float32 rows out.
    """

    def __init__(self, ns, stage_name, out_dir):
        self.name = stage_name

        # --! the worker needs the model and its weights, not the data
        args = copy.copy(ns)
        args.data, args.traj_nsample = None, None

        context = multiprocessing.get_context('spawn')
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(target=_serve_stage, args=(child_conn, args, stage_name, out_dir), daemon=True)
        self._process.start()
        child_conn.close()
        self._receive()

    def _receive(self):
        try:
            status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError(f'worker process of quantized stage {self.name} died')
        if status == 'error':
            raise RuntimeError(f'quantized stage {self.name}: {payload}')
        return payload

    def __call__(self, x):
        self._conn.send(np.ascontiguousarray(x, dtype=np.float32))
        return self._receive()

    def close(self):
        try:
            self._conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self._process.join(timeout=10)


class torch_process_stage(torch.nn.Module):
    """
    A process_stage as a torch module, so that quantized stages can replace the stages of a composite torch model.
    """

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return torch.from_numpy(self.stage(x.detach().cpu().numpy()))


class stage_recorder:
    """
    Remembers the order in which stages are called.
    """

    def __init__(self, name, stage, log):
        self.name, self.stage, self.log = name, stage, log

    def __call__(self, x):
        self.log.append(self.name)
        return self.stage(x)


def make_input_sets(model, trajectories):
    """
    Lets ``model`` turn raw ``trajectories`` into ``model`` inputs.
    The resulting ``model`` inputs are then split per trajectory to form a tuple of single-trajectory model inputs.

    Required parameters:
        model           torch model with a method ``make_inputs(trajectories)``
        trajectories    3d torch tensor, trajectories with shape [ntraj, traj_nsample, nchan]

    Returns:
        input_sets      list of tuples, every tuple holds a single model parameter shaped as [npast, past_nsample, nchan]
    """
    if not hasattr(model, 'make_inputs'):
        raise SystemExit(
            '[error]: the model has no make_inputs(trajectories) method, so --data cannot be turned into model inputs')

    try:
        inputs = model.make_inputs(trajectories)
    except TypeError as e:
        raise SystemExit(f'[error]: model.make_inputs(trajectories) failed: {e}\n[note]: it must be an instance method taking only a [ntraj, traj_nsample, nchan] tensor')

    # --! ensure inputs are a tuple
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)

    # --! ensure returned inputs have the leading dimension equal to the number of trajectories
    ntraj = trajectories.shape[0]
    for x in inputs:
        if x.shape[0] != ntraj:
            raise SystemExit(
                f'[error]: make_inputs must return one entry per trajectory (leading dimension {ntraj}), but got shape {tuple(x.shape)}')

    # --! return a list of single-element tuples
    #
    # --! the idea of using single-element tuples is to allow putting more model parameters into the tuples,
    # --! if such need comes
    return [tuple(x[t] for x in inputs) for t in range(ntraj)]


def check_weights(model, path):
    """
    Checks if the keys of a state_dict file located at ``path`` match given ``model``.
    This method uses non-strict load.

    Required parameters:
        model       torch model
        path        string, a path to a .pth file with weights

    Returns:
        status      dictionary, a check status summarizing missing or unexpected keys (if any)
    """
    state = torch.load(path, map_location='cpu')
    result = model.load_state_dict(state, strict=False)
    return {
        'file'              : path,
        'missing_keys'      : list(result.missing_keys),
        'unexpected_keys'   : list(result.unexpected_keys)
    }


def build_model(module, config):
    """
    Instantiates a model imported from ``module`` using parameters from ``config``.
    Parameters of the model constructor are processed as follows:
        missing required parameters     error
        unknown keys                    reported
        silently used defaults          reported
    
    Required parameters:
        module      imported module, the module is expected to define a class named 'model'
        config      dictionary, configuration read from a .json file

    Returns:
        model                   torch model, a model instantiated using constructor parameters from ``config``
        construction_report     dictionary, a report summarizing keywords, unknown keys, defaults, etc.
    """
    # --! retrieve the signature of model constructor parameters
    model = module.model
    params = [p for p in inspect.signature(model.__init__).parameters.values() if p.name != 'self']

    # --! inspect parameter kinds in model constructor
    P = inspect.Parameter
    takes_any = any(p.kind == P.VAR_KEYWORD for p in params)
    named = [p for p in params if p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)]
    names = [p.name for p in named]

    # --! gather parameter deviations
    missing = [p.name for p in named if p.default is P.empty and p.name not in config]
    unknown = [] if takes_any else [k for k in config if k not in names]
    defaults_used = {p.name: p.default for p in named if p.default is not P.empty and p.name not in config}

    # --! report parameter deviations
    if missing:
        raise SystemExit(f"[error]: config lacks required constructor parameters: {missing}\n[note]: constructor expects: {names}")
    if unknown:
        print(f'[warn]: config keys not accepted by constructor and ignored: {unknown}')
    if defaults_used:
        print(f'[note]: constructor defaults used for: {defaults_used}')

    # --! build key-value model arguments
    kwargs = {k: v for k, v in config.items() if k not in unknown}

    # --! compile a construction report with model build details
    report = {
        'style': 'keywords',
        'config': config,
        'unknown_keys': unknown,
        'defaults_used': defaults_used
    }

    # --! instantiate the model and return it together with the construction report
    return model(**kwargs), report


def get_stage_names(model):
    """
    Gets names of ``model`` submodules (stages).

    Required parameters:
        model       torch model, a composite model with submodules (stages)

    Returns:
        names       list of strings, names of model stages
    """
    return [name for name, _ in model.named_children()]


def capture_stage_inputs(model, stage, input_sets):
    """
    Runs ``model`` over ``input_sets`` to capture everything ``stage`` receives as input.

    Required parameters:
        model       torch model, a composite model with submodules (stages)
        stage       string, name of a submodule in ``model``
        input_sets  list of tuples, every tuple holds a single model parameter shaped as [npast, past_nsample, nchan]

    Returns:
        captured    torch tensor, captured stage inputs concatenated along dimension 0
    """
    captured = []
    handle = getattr(model, stage).register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            for inputs in input_sets:
                model(*inputs)
    finally:
        handle.remove()
    if not captured:
        raise SystemExit(f'[error]: stage {stage} was never called by model')
    return torch.cat(captured, dim=0)


def capture_first_rows(model, stages, input_sets):
    """
    Captures first batch-1 row each stage receives in one ``model`` run.

    Required parameters:
        model       torch model, a composite model with stages (submodules)
        stages      list of strings, stage names obtained using ``get_stage_names``
        input_sets  list of tuples, every tuple holds a single model parameter shaped as [npast, past_nsample, nchan]

    Returns:
        first       dict, per-stage data samples that stages receive first
    """
    first, handles = {}, []

    def make_hook(name):
        def hook(module, args):
            if name not in first:

                # --! args is a tuple of forward parameters;
                # --! since our forward receives only one parameter, we have a single-element tuple; hence args[0] below
                #
                # --! first tensor dimension is a batch dimension, so [:1] below extracts the first element from the batch
                first[name] = args[0][:1].detach().clone()
        return hook

    # --! capture first rows from first input tuple
    for name in stages:
        handles.append(getattr(model, name).register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(*input_sets[0]) # use the first tuple from the list
    finally:
        for h in handles:
            h.remove()
    return first


def extract_output(model_output, field):
    """
    Extracts specified ``field`` from ``model_output``.

    Required parameters:
        model_output    named tuple, composite output from model
        field           string, name of output field

    Returns:
        output_field    torch tensor, extracted output field
    """

    # --! in case model output is already a tensor, return the tensor
    if isinstance(model_output, torch.Tensor):
        return model_output

    # --! otherwise treat the output as a named tuple or a data class and look for a properly named attribute
    try:
        return getattr(model_output, field)
    except AttributeError:
        raise SystemExit(f'[error]: model output has no field {field}; use --out-field to specify the right name')


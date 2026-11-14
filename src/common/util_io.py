import importlib.util
import numpy as np
import argparse
import json


def read_trajectories(path, traj_nsample, delimiter=','):
    """
    Reads trajectories from a .csv file located at ``path``.
    The .csv file contents are expected to be shaped as [ntraj * traj_nsample, nchan].

    Required parameters:
        path            string, path of read file
        traj_nsample    int, number of samples in a trajectory

    Optional parameters:
        delimiter       string, a delimiter between file columns, default=','

    Returns:
        trajectories    numpy array of float32 values, a 3d numpy array with read trajectories shaped as [ntraj, traj_nsample, nchan]
    """
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float32, ndmin=2)
    if data.shape[0] % traj_nsample != 0:
        raise SystemExit(f'[error]: {path} has {data.shape[0]} rows, not a multiple of --traj-nsample {traj_nsample}')

    ntraj = data.shape[0] // traj_nsample
    return data.reshape(ntraj, traj_nsample, data.shape[1])


def read_arguments(description='bridge: pipeline from ai to fpga', base_used=True, extra_args={}, extra_options={}):
    """
    Reads arguments from command line.
    User can provide ``extra_args`` and ``extra_options`` which represent extra argumenst and double-dashed options
    that are added to base arguments. The extra adhere to the syntax used by add_argument method from argparse.
    User can disable base arguments and options completely by setting ``base_used`` to False. In this case,
    the user must provide the extra arguments (the extra options may be missing).

    Optional parameters:
        description     string, description of current app, default='bridge: pipeline from ai to fpga'
        base_used       bool, whether to use base arguments and options, default=True
        extra_args      dict of dicts, arguments with fields following add_argument interface from argparse, default=empty dict
        extra_options   dict of dicts, options with fields following add_argument interface from argparse, default=empty dict

    Returns:
        ns              namespace, arguments shared by all scripts that run model on data
    """

    if not base_used and not extra_args:
        raise SystemExit('[error]: for arguments reading there must either be base or extra arguments specified')

    ap = argparse.ArgumentParser(description=description)

    # --! specify base arguments
    if base_used:
        ap.add_argument('model', help="python file defining class 'model'")
        ap.add_argument('config', help='model_config.json (constructor parameters)')

    if extra_args:
        for arg_name, arg_kwargs in extra_args.items():
            ap.add_argument(arg_name, **arg_kwargs)

    # --! specify base options
    if base_used:
        ap.add_argument('--weights', help='state_dict (.pth) of the whole composite model')
        ap.add_argument('--data', help='csv with raw trajectories (rows = time, columns = channels)')
        ap.add_argument('--traj-nsample', type=int, help='number of rows per trajectory in --data')
        ap.add_argument(
            '--out-quant',
            help='root quantization output directory; every stage gets its own subdirectory',
            default='quantize_result')

    # --! check and add extra options to argument parser
    if extra_options:
        for option_name, option_kwargs in extra_options.items():

            # --! options must start with a double-dash
            if not option_name.startswith('--'):
                raise SystemExit(f"[error]: option names must start with '--', but got '{option_name}'")

            ap.add_argument(option_name, **option_kwargs)

    return ap.parse_args()


def read_config(path):
    """
    Reads configuration from a .json file located at ``path``.

    Required parameters:
        path        string, path to a .json file with configuration

    Returns:
        config      dict, read configuration
    """
    with open(path) as f:
        config = json.load(f)
    return config


def write_plan(plan, path='model_plan.json'):
    """
    Writes model ``plan`` to a .json file located at ``path``.

    Required parameters:
        plan    dict, model plan produced by model analysis

    Optional parameters:
        path    string, path to saved .json file, default='model_plan.json'
    """
    with open(path, 'w') as f:
        json.dump(plan, f, indent=2, default=str)

    print(f'[note]: plan written to {path}')


def tabulate(headers, rows):

    # --! compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # --! build format string
    fmt = "    ".join(f"{{:<{w}}}" for w in widths)
    sep = "-  -".join("-" * w for w in widths)

    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))

    return '\n'.join(lines)


def import_module(path, name, attributes):
    """
    Imports a module under given ``name`` located at ``path``. The module is checked to define ``attributes``.

    Required parameters:
        path            string, a path to imported module
        name            string, name of module
        attributes      list of strings, names of classes, functions, etc. defined in imported module

    Returns:
        module          imported module
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in attributes:
        if not hasattr(module, attr):
            raise SystemExit(f"[error]: {path} defines no attribute named {attr}")

    return module


def read_json(path):
    """
    Reads a .json file located at ``path``.

    Required parameters:
        path        string, path to a .json file to be read

    Returns:
        contents    dict, read file contents for matted as a dictionary
    """
    with open(path) as f:
        return json.load(f)


def write_json(data, path):
    """
    Writes ``data`` to a .json file located at ``path``.

    Required paraneters:
        data        dict, written file contents formatted as a dictionary
        path        string, path to a .json file to be written
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)



import ast
import inspect
import textwrap
from dataclasses import dataclass, asdict

import numpy as np
import torch

import util
import util_io
import util_model


# ---------------------------------------------------------------------------
# 1. Source analysis (static, torch-free)
# ---------------------------------------------------------------------------

@dataclass
class Call:
    kind: str          # 'stage' | 'helper' | 'op' | 'expr'
    name: str          # 'encoder' | 'make_windows' | 'torch.flatten' | source of an expression
    assigned_to: list  # variables the result is stored in, if any
    uses: list         # variable names appearing in the arguments
    in_lambda: bool    # True if the call sits inside a lambda
    lineno: int


_IGNORED_NAMES = {'self', 'torch', 'np', 'numpy', 'F'}


def _dotted(node):
    """ast node of `a.b.c` -> 'a.b.c' (None if it is not a plain dotted name)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


class _CallCollector(ast.NodeVisitor):

    def __init__(self, source, line_offset, stage_names, helper_names):
        self.source = source
        self.line_offset = line_offset
        self.stage_names = set(stage_names)
        self.helper_names = set(helper_names)
        self.calls = []
        self._assigned = {}      # id(call node) -> target names
        self._lambda_depth = 0

    def _value_source(self, stmt):
        """Source text of an assignment's right-hand side (best effort on Python 3.7)."""
        if hasattr(ast, 'get_source_segment'):             # Python 3.8+
            segment = ast.get_source_segment(self.source, stmt.value)
            if segment:
                return segment
        # Python 3.7 has no end positions: take the statement's source lines
        last = max(getattr(n, 'lineno', stmt.lineno) for n in ast.walk(stmt))
        text = ' '.join(line.strip()
                        for line in self.source.splitlines()[stmt.lineno - 1:last])
        return text.split('=', 1)[1].strip() if '=' in text else text

    def _uses(self, nodes):
        """Variables and self.<attribute> constants read inside `nodes`."""
        lambda_params = {a.arg for n in nodes for m in ast.walk(n)
                         if isinstance(m, ast.Lambda) for a in m.args.args}
        uses = set()
        for n in nodes:
            for m in ast.walk(n):
                if isinstance(m, ast.Name) and m.id not in _IGNORED_NAMES | lambda_params:
                    uses.add(m.id)
                elif (isinstance(m, ast.Attribute) and isinstance(m.value, ast.Name)
                      and m.value.id == 'self'
                      and m.attr not in self.stage_names | self.helper_names):
                    uses.add('self.' + m.attr)
        return sorted(uses)

    def visit_Assign(self, node):
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node.value, ast.Call):
            self._assigned[id(node.value)] = targets
        self.generic_visit(node)
        # plain arithmetic (e.g. normalization) is not a call, so record it here
        if isinstance(node.value, (ast.BinOp, ast.UnaryOp)):
            self.calls.append(Call(
                kind='expr',
                name=self._value_source(node),
                assigned_to=targets,
                uses=self._uses([node.value]),
                in_lambda=self._lambda_depth > 0,
                lineno=node.lineno + self.line_offset))

    def visit_Lambda(self, node):
        self._lambda_depth += 1
        self.generic_visit(node)
        self._lambda_depth -= 1

    def visit_Call(self, node):
        # arguments are evaluated before the call itself, so visit them first;
        # this makes self.calls follow execution order
        self.generic_visit(node)

        name = _dotted(node.func)
        if name is None:
            return

        kind = 'op'
        if name.startswith('self.'):
            attr = name[len('self.'):]
            if attr in self.stage_names:
                kind, name = 'stage', attr
            elif attr in self.helper_names:
                kind, name = 'helper', attr

        arg_nodes = [*node.args, *[k.value for k in node.keywords]]
        self.calls.append(Call(
            kind=kind,
            name=name,
            assigned_to=self._assigned.get(id(node), []),
            uses=self._uses(arg_nodes),
            in_lambda=self._lambda_depth > 0,
            lineno=node.lineno + self.line_offset))


def scan_function(fn, stage_names=(), helper_names=()):
    """Calls made by function `fn`, in execution order (line numbers are file lines)."""
    lines, first_line = inspect.getsourcelines(fn)
    source = textwrap.dedent(''.join(lines))
    collector = _CallCollector(source, first_line - 1, stage_names, helper_names)
    collector.visit(ast.parse(source))
    return collector.calls


def scan_forward(model):
    """
    Static analysis of `model.forward` plus the helper methods it calls.
    Returns {'forward': [Call, ...], 'helpers': {name: [Call, ...]}}.
    """
    cls = type(model)
    stage_names = [name for name, _ in model.named_children()]
    # only functions defined in the class itself, not inherited from nn.Module
    helper_names = [k for k, v in vars(cls).items()
                    if inspect.isfunction(v) and k not in ('forward', '__init__')]

    forward_calls = scan_function(cls.forward, stage_names, helper_names)

    helpers = {}
    for call in forward_calls:
        if call.kind == 'helper' and call.name not in helpers:
            helpers[call.name] = scan_function(
                vars(cls)[call.name], stage_names, helper_names)
    return {'forward': forward_calls, 'helpers': helpers}


# ---------------------------------------------------------------------------
# 2. Structure (runtime)
# ---------------------------------------------------------------------------

def describe_stages(model):
    """Per child module: layer types, plus functional ops used in its forward."""
    stages = {}
    for name, child in model.named_children():
        leaves = [type(m).__name__ for m in child.modules()
                  if not list(m.children())]
        functional = [c.name for c in scan_function(type(child).forward)
                      if c.kind == 'op' and not c.name.startswith('self.')]
        stages[name] = {
            'class': type(child).__name__,
            'layers': leaves,
            'functional_ops': functional,   # e.g. torch.mean inside forward
        }
    return stages


def find_constants(model):
    """
    Non-module state the CPU glue needs to know about.
    A plain tensor attribute is NOT saved in state_dict (use register_buffer
    for that), so it will not travel to the board together with the weights.
    """
    plain_tensors, config = {}, {}
    for key, value in vars(model).items():
        if key.startswith('_') or key == 'training':
            continue
        if isinstance(value, torch.Tensor):
            plain_tensors[key] = {'shape': list(value.shape),
                                  'value': value.tolist() if value.numel() <= 8 else '...'}
        elif isinstance(value, (bool, int, float, str)):
            config[key] = value
    return plain_tensors, config


# ---------------------------------------------------------------------------
# 3. Behaviour (dynamic, via forward hooks)
# ---------------------------------------------------------------------------

def _summarize_calls(calls, child_names):
    """records per stage -> one summary per stage (pure python, easy to test)"""
    summary = {}
    for name in child_names:
        recs = calls.get(name, [])
        if not recs:
            summary[name] = {'ncalls': 0}
            continue
        summary[name] = {
            'ncalls': len(recs),
            'in_shapes': [list(sh) for sh in sorted({tuple(r['in_shape']) for r in recs})],
            'in_dtype': recs[0]['in_dtype'],
            'in_range': [min(r['in_min'] for r in recs), max(r['in_max'] for r in recs)],
            'out_shapes': [list(sh) for sh in sorted({tuple(r['out_shape']) for r in recs})],
            'out_range': [min(r['out_min'] for r in recs), max(r['out_max'] for r in recs)],
        }
    return summary


def trace_stages(model, input_sets):
    """
    Runs ``model`` once per entry of ``input_sets`` and summarizes every call of every child module (stage).

    Required parameters:
        model       callable model, a model derived from torch.nn.Module with a forward method that accepts inputs from ``input_sets``
        input_sets  list of tuple, a list of tuples with model inputs, one model input per trajectory

    Returns:
        summary     dictionary, a summary with data shapes, number of calls, ranges, types, etc.
    """
    calls, handles = {}, []

    # --! helper function for hook setting
    def make_hook(name):
        def hook(module, args, output):
            x, y = args[0].detach(), output.detach()
            calls.setdefault(name, []).append({
                'in_shape': list(x.shape), 'in_dtype': str(x.dtype),
                'in_min': float(x.min()), 'in_max': float(x.max()),
                'out_shape': list(y.shape),
                'out_min': float(y.min()), 'out_max': float(y.max()),
            })
        return hook

    # --! set hooks on forward calls inside named children of a composite model
    child_names = [name for name, _ in model.named_children()]
    for name, child in model.named_children():
        handles.append(child.register_forward_hook(make_hook(name)))

    # --! with hooks set, call the model with every input from the given input set
    try:
        model.eval()
        with torch.no_grad():
            for t, inputs in enumerate(input_sets):
                try:
                    model(*inputs)
                except Exception as e:
                    raise SystemExit(
                        f'[error]: model failed on trajectory {t} with input shapes {[tuple(x.shape) for x in inputs]} due to error {type(e).__name__}: {e}')

    # --! when finished, remove forward hooks
    finally:
        for h in handles:
            h.remove()

    return _summarize_calls(calls, child_names)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def analyze(model, input_sets=()):
    source = scan_forward(model)
    plain_tensors, config = find_constants(model)
    plan = {
        'stages': describe_stages(model),
        'pipeline': [asdict(c) for c in source['forward']],
        'helpers': {k: [asdict(c) for c in v] for k, v in source['helpers'].items()},
        'buffers': {k: list(v.shape) for k, v in model.named_buffers()},
        'plain_tensor_attributes': plain_tensors,
        'config_attributes': config,
    }
    if input_sets:
        plan['trace'] = trace_stages(model, input_sets)
        plan['trace_ntraj'] = len(input_sets)
    return plan


def print_report(plan):
    if 'construction' in plan:
        print('CONSTRUCTION  (%s)' % plan['construction']['style'])
        print('  config: %s' % plan['construction']['config'])

    print('\nSTAGES (DPU candidates = child modules)')
    for name, info in plan['stages'].items():
        print(f"  {name}: {info['class']}  layers={info['layers']}  "
              f"functional_ops={info['functional_ops']}")

    print('\nFORWARD, in execution order')
    for c in plan['pipeline']:
        where = '  [inside lambda]' if c['in_lambda'] else ''
        out = ','.join(c['assigned_to']) + ' = ' if c['assigned_to'] else ''
        shown = c['name'] if c['kind'] == 'expr' else c['name'] + '(...)'
        if c['kind'] == 'expr' and out and shown.startswith(out.split(' = ')[0]):
            out = ''
        print(f"  line {c['lineno']:>3}  {c['kind']:<6} {out}{shown}"
              f"   <- {', '.join(c['uses'])}{where}")

    for name, calls in plan['helpers'].items():
        print(f'\nHELPER {name}()')
        for c in calls:
            print(f"  line {c['lineno']:>3}  {c['kind']:<6} {c['name']}")

    print('\nNON-MODULE STATE')
    for k, v in plan['buffers'].items():
        print(f'  buffer {k} {v}   (saved in state_dict)')
    for k, v in plan['plain_tensor_attributes'].items():
        print(f'  tensor {k} {v["shape"]} = {v["value"]}   '
              f'(NOT in state_dict: use register_buffer)')
    for k, v in plan['config_attributes'].items():
        print(f'  config {k} = {v}')

    if 'weights_check' in plan:
        w = plan['weights_check']
        print('\nWEIGHTS  (%s)' % w['file'])
        print('  missing keys   :', w['missing_keys'] or 'none')
        print('  unexpected keys:', w['unexpected_keys'] or 'none')
    else:
        print('[note]: no weights check; pass --weights to enable it')

    if 'trace' in plan:
        print('\nOBSERVED STAGE CALLS  (%d trajectories)' % plan['trace_ntraj'])
        for name, r in plan['trace'].items():
            if r['ncalls'] == 0:
                print(f'  {name}: NEVER CALLED')
                continue
            lo, hi = r['in_range']
            olo, ohi = r['out_range']
            print(f"  {name}: {r['ncalls']} calls, in {r['in_shapes']} {r['in_dtype']} "
                  f"range [{lo:.3g}, {hi:.3g}] -> out {r['out_shapes']} range [{olo:.3g}, {ohi:.3g}]")
    else:
        print('[note]: no behaviour trace; pass --data and --traj-nsample to enable it')


def main():
    options = {
        '--out-plan': {
            'help'    : 'path to a .json file where the plan will be saved, default=model_plan.json',
            'default' : 'model_plan.json'
        }
    }
    ns = util_io.read_arguments(
        description='analyze the structure and dynamic behavior of a composite model', extra_options=options)

    # --! load current session
    model, construction, weights_check, trajectories, input_sets = util.load_session(ns, strict=False)    # a weights mismatch is reported in the plan, not fatal

    if ns.data:
        print(f'[note]: read {trajectories.shape[0]} trajectories of shape {tuple(trajectories.shape[1:])} from {ns.data}')

    plan = analyze(model, input_sets)
    plan['construction'] = construction
    if ns.weights:
        plan['weights_check'] = weights_check

    print_report(plan)
    util_io.write_plan(plan, ns.out_plan)


if __name__ == '__main__':
    main()


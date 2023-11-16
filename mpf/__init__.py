from io import FileIO
import os
import inspect
import yaml
import subprocess
import argparse
import shutil
import ipyparallel as ipp
import pandas as pd
import random
import logging
import math
from tqdm.auto import tqdm

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, List, Tuple, Callable, Union
from time import sleep

import mpf.wsp as wsp

run_logger = logging.getLogger('run')
run_logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(asctime)s [%(role)s] %(function)s: %(message)s'))
if not run_logger.hasHandlers():
    run_logger.addHandler(sh)

RESERVED_VARIABLES = {'mpf_ctx'}
random.seed('mpf')

variables: Dict[str, 'Variable'] = {}
functions: List[Tuple[str, int, Callable]] = []
init_functions: List[Tuple[str, Callable]] = []
helpers: List[Callable] = []
roles: Dict[str, 'Role'] = {}
experiment_globals: Dict[str, Any] = {}
wsp_points: List[List[float]] = []

@dataclass
class Variable():
    """ A variable to explore in an experiment.

    >>> v = Variable('a', list(range(8)))
    >>> list(v)
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    name: str
    values: List[Any]

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def explore(variables: List['Variable']):
        """ Yields dictionaries with a value within the range of each variables.

        >>> v1 = Variable('a', [1, 2])
        >>> v2 = Variable('b', ['x', 'y'])
        >>> list(Variable.explore([v1, v2]))
        [{'a': 1, 'b': 'x'}, {'a': 1, 'b': 'y'}, {'a': 2, 'b': 'x'}, {'a': 2, 'b': 'y'}]
        >>> v3 = WSPVariable('c', values=[], range=[4, 8], index=0)
        >>> wsp_points.clear()
        >>> wsp_points.extend([[0], [0.5]])
        >>> list(Variable.explore([v1, v2, v3]))
        [{'a': 1, 'b': 'x', 'c': 4}, {'a': 1, 'b': 'x', 'c': 6.0}, {'a': 1, 'b': 'y', 'c': 4}, {'a': 1, 'b': 'y', 'c': 6.0}, {'a': 2, 'b': 'x', 'c': 4}, {'a': 2, 'b': 'x', 'c': 6.0}, {'a': 2, 'b': 'y', 'c': 4}, {'a': 2, 'b': 'y', 'c': 6.0}]
        """
        def wsp_values(wsp_variables: List[WSPVariable]) -> List[Dict[str, Any]]:
            return [{wsp_variables[i].name: v for i, v in enumerate(values)} for values in zip(*wsp_variables)]

        def explore_variables(variables: List[Variable], wsp_variables: List[WSPVariable]):
            if variables:
                variable = variables[0]
                for value in variable:
                    if len(variables) > 1:
                        for v in explore_variables(variables[1:], wsp_variables):
                            yield {variable.name: value} | v
                    elif wsp_variables:
                        for wsp_vals in wsp_values(wsp_variables):
                            yield {variable.name: value} | wsp_vals
                    else:
                        yield {variable.name: value}
            
        if any(type(v) is not WSPVariable for v in variables):
            yield from explore_variables([v for v in variables if type(v) is not WSPVariable], [v for v in variables if type(v) is WSPVariable])
        else:
            yield from wsp_values(variables)

@dataclass
class WSPVariable(Variable):
    """ A variable to explore based on the WSP space filling algorithm in an experiment. 
    
    >>> v1 = WSPVariable('a', values=[], range=[4, 8], index=0)
    >>> wsp_points.clear()
    >>> wsp_points.extend([[0, 1], [0.5, 0.01], [1, 0.33]])
    >>> list(v1)
    [4, 6.0, 8]
    >>> v2 = WSPVariable('b', values=['A', 'B', 'C', 'D'], index=1)
    >>> list(v2)
    ['D', 'A', 'B']
    """
    range: List[Union[int, float]] = field(default_factory=list)
    index: int = field(default_factory=count().__next__)

    def __iter__(self):
        assert wsp_points, f"No WSP points, please set wsp_target= in run_experiment()"
        assert len(wsp_points[0]) > self.index, "Not enough dimensions in wsp_points"
        wsp_values = [v[self.index] for v in wsp_points]
        if self.range:
            return iter([self.range[0] + (self.range[1] - self.range[0]) * v for v in wsp_values])
        else:
            return iter([list(self.values)[int(math.ceil(v * len(self.values)) - 1)] for v in wsp_values])


@dataclass
class Role():
    """ A role defining what code to run, when to run it 
        and the network interfaces of the underlying machine
    """
    name: str
    functions: List[Tuple[int, Any]]  # A list of tuple of delay value and IPython function to execute
    interfaces: List[Tuple[str, str]]  # A list of tuple interface name, ip address

def create_profile(profile_dir: str, cluster: dict):
    """ Populates the given directory with a blank IPython profile structure, enables SSH 
        and sets up the IPython engines based on the given YAML cluster file.
        Creates the roles available from the cluster definition.
    """
    engines = {}
    for machine_spec in cluster['machines']:
        engine = machine_spec['hostname']
        if 'user' in machine_spec:
            engine = f"{machine_spec['user']}@{engine}"
        engines[engine] = 1  # Starts one engine per host

    # The controller is the ipyparallel controller listening for external connections
    controller_node = cluster['controller']['hostname']
    controller_ip = cluster['controller']['control_ip']
    controller_ports = cluster['controller']['ports']

    shutil.rmtree(profile_dir, ignore_errors=True)
    p = subprocess.run(['ipython', 'profile', 'create', '--parallel', f'--profile-dir={profile_dir}'])
    assert p.returncode == 0

    with open(os.path.join(profile_dir, 'ipcluster_config.py'), 'a') as config:
        config.write("""
c.Cluster.engine_launcher_class = 'ssh'
c.SSHEngineSetLauncher.engines = {engines}
c.SSHEngineSetLauncher.remote_profile_dir = '/tmp/mpf-ipy-profile'
c.Cluster.controller_launcher_class = 'ssh'
c.SSHControllerLauncher.location = '{controller_node}'
c.SSHControllerLauncher.controller_cmd = ['{python_path}', '-m', 'ipyparallel.controller', '--ip={controller_ip}']
c.SSHLauncher.remote_python = '{python_path}'"""
    .format(engines=repr(engines), controller_node=controller_node, controller_ip=controller_ip, python_path=cluster['global']['python_path']))

    with open(os.path.join(profile_dir, 'ipcontroller_config.py'), 'a') as config:
        config.write("c.IPController.ports = {}".format(repr(controller_ports)))

    with open(os.path.join(profile_dir, 'startup', '00-mpf_exec.ipy'), 'w') as mpf_exec_file:
        mpf_exec_file.write("""
from IPython.core.magic import register_line_magic
import shlex
def ex(*args):
    global mpf_log
    line = ' '.join(args)
    if any(s == '&' for s in shlex.shlex(line)):
        out = []
        !$line
    else:
        out = !$line
    mpf_log.append((line, out))
    return out
register_line_magic(ex)
del ex
        """.format(python_path=cluster['global']['python_path']))

    for e in engines:
        p = subprocess.run(['rsync', '--mkpath', os.path.join(profile_dir, 'startup', '00-mpf_exec.ipy'), f'{e}:/tmp/mpf-ipy-profile/startup/00-mpf_exec.ipy'])
        assert p.returncode == 0

    return controller_node

def add_variable(name: str, values):
    """ Adds the given variable and values to explore in the experiment. """
    assert name not in variables, f"variable {name} already exists"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    if type(values) is range:
        values = list(values)
    variables[name] = Variable(name, values)

def add_wsp_variable(name: str, values=None, range=None):
    """ Adds the given variable and delegates the choice of its values the WSP space filling algorithm.
        To provide a fixed set of values that will be explored, use the values argument.
        To provide a range from which values will be sampled by WSP, use the range argument.
    """
    assert name not in variables, f"variable {name} already exists"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    assert values is not None or range is not None, "One of values and range must be not None"
    variables[name] = WSPVariable(name, values=values or [], range=range or [])

def register_globals(**kwargs):
    """ Updates the experiment global variables with the given names and values. """
    experiment_globals.update(kwargs)

def run(role: str='main', delay: int=0):
    """ Registers the given function to be executed by a role at given time as part of the experiment. """
    assert role in roles, f"role {role} is not defined in the cluster"
    def inner(func):
        functions.append((role, delay, func))
        return func
    return inner

def init(role: str='main'):
    """ Registers the given function to be executed before the start of the experiment. """
    assert role in roles, f"role {role} is not defined in the cluster"
    def inner(func):
        init_functions.append((role, func))
        return func
    return inner

def helper():
    """ Registers the given function as an helper to be pushed on engines. """
    def inner(func):
        helpers.append(func)
        return func
    return inner

def run_experiment(n_runs=3, wsp_target=None, log_ex=False):
    """ Runs the experiment and returns the results gathered. """
    if log_ex:
        run_logger.setLevel(logging.INFO)
    if wsp_target is not None:
        ps = wsp.PointSet.from_random(wsp_target * 100, sum([type(v) is WSPVariable for v in variables.values()]), 'mpf')
        ps.adaptive_wsp(wsp_target)
        wsp_points.extend(ps.get_remaining())
    results = []
    variable_values = []
    experiments = list(Variable.explore(list(variables.values()))) * n_runs
    random.shuffle(experiments)
    for experiment_values in tqdm(experiments):
        row = {}
        for role, delay, function in functions:
            role_id = list(roles).index(role)
            client[role_id].push(dict(mpf_log=[], **{f.__name__: f for f in helpers}), block=True)
            sleep(delay)
            mpf_ctx = {'roles': {r: {'interfaces': roles[r].interfaces} for r in roles}, 'role': role}
            function_args = inspect.getfullargspec(function).args
            call_args = {arg_name: experiment_values[arg_name] for arg_name in function_args if arg_name not in RESERVED_VARIABLES}
            if 'mpf_ctx' in function_args:
                call_args['mpf_ctx'] = mpf_ctx
            client[role_id].push(experiment_globals)
            result = client[role_id].apply_sync(function, **call_args)
            if result is None:
                result = {}
            assert type(result) is dict, "return value of @mpf.run functions should be a dict with the results names and values or None"
            assert all(k not in row for k in result.keys()), f"function {function} returned a result name that conflicts with experiment value"
            mpf_log = client[role_id].pull('mpf_log', block=True)
            for line, out in mpf_log:
                run_logger.info('\n'.join([line] + out), extra={'function': function.__name__, 'role': role})
            row.update(result)
        results.append(row)
        variable_values.append(tuple(experiment_values.values()))
        client.abort()
    index = pd.MultiIndex.from_tuples(variable_values, names=[v.name for v in variables.values()])
    return pd.DataFrame(results, index=index)

def start_cluster_and_connect_client(cluster_file: FileIO):
    cluster_profile = f"profile_{os.path.basename(cluster_file.name)}"
    cluster_definition = yaml.safe_load(cluster_file)
    for machine_spec in cluster_definition['machines']:
        assert machine_spec['role'] not in roles, f"role {machine_spec['role']} already exists"
        roles[machine_spec['role']] = Role(machine_spec['role'], [], machine_spec['interfaces'])
    controller_node = cluster_definition['machines'][0]['hostname']
    try:
        cluster = ipp.Cluster.from_file(profile_dir=cluster_profile)
    except FileNotFoundError:
        create_profile(cluster_profile, cluster_definition)
        subprocess.run(['ipcluster', 'start', f'--profile-dir={cluster_profile}', '--daemonize=True', '--log-level=ERROR'])
        cluster = ipp.Cluster.from_file(profile_dir=cluster_profile)
    return cluster.connect_client_sync(sshserver=controller_node)

if __name__ == "mpf":
    parser = argparse.ArgumentParser(description='mpf experiment')
    parser.add_argument('-c', '--cluster', metavar='cluster.yaml', type=argparse.FileType('r'), required=True,help='The YAML file describing the cluster')
    args = parser.parse_args()

    if args.cluster:
        client = start_cluster_and_connect_client(args.cluster)
        client.wait_for_engines(timeout=10, block=True)

from io import FileIO
import os
import sys
import inspect
import string
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

from dataclasses import dataclass, field, asdict
from itertools import count
from typing import Any, Dict, Iterator, List, Optional, Tuple, Callable, Union
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

setup_done: bool = False
variables: Dict[str, 'Variable'] = {}
functions: List[Tuple[Optional[str], Optional[str], int, bool, Callable]] = []
init_functions: List[Tuple[str, Callable]] = []
helpers: List[Callable] = []
roles: Dict[str, 'Role'] = {}
links: Dict[str, 'Link'] = {}
experiment_globals: Dict[str, Any] = {}
wsp_points: List[List[float]] = []
engines: List[str] = []

experiment_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
random.seed('mpf')

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
class LinkInterface():
    """ An interface part of a link. """
    name: str
    ip: Optional[str]
    role: str  # The role owning this interface
    link: Optional[str]  # The link composed of this interface
    direction: str  # Either forward or backward
    neighbour: Optional[str]  # The role connected to the other end of this interface, when it is known


@dataclass
class Role():
    """ A role defining what code to run, when to run it 
        and the network interfaces of the underlying machine
    """
    name: str
    machine_id: int
    namespace: Optional[str]
    cpu_id: Optional[int]
    functions: List[Tuple[int, bool, Any]]  # A list of tuple of delay value, parallel mode, and IPython function to execute
    interfaces: List[LinkInterface]


@dataclass
class Link():
    """ A network link composed of one or two interfaces belonging
        to two roles. When the link is unidirectional, only the forward
        interface is set.
    """
    name: str
    forward: Optional[LinkInterface]
    backward: Optional[LinkInterface]

    def __iter__(self):
        return iter([self.forward, self.backward] if self.backward else [self.forward])

def create_profile(profile_dir: str, cluster: dict):
    """ Populates the given directory with a blank IPython profile structure, enables SSH 
        and sets up the IPython engines based on the given YAML cluster file.
        Creates the roles available from the cluster definition.
    """
    # The controller is the ipyparallel controller listening for external connections
    controller_node = f"{cluster['controller']['user']}@{cluster['controller']['hostname']}"
    controller_ip = cluster['controller']['control_ip']
    controller_ports = cluster['controller']['ports']

    shutil.rmtree(profile_dir, ignore_errors=True)
    p = subprocess.run(['ipython', 'profile', 'create', '--parallel', f'--profile-dir={profile_dir}'])
    assert p.returncode == 0

    with open(os.path.join(profile_dir, 'ipcluster_config.py'), 'a') as config:
        config.write("""
c.Cluster.n = {n}
c.Cluster.engine_launcher_class = 'ssh'
c.SSHEngineSetLauncher.engines = {engines}
c.SSHEngineSetLauncher.remote_profile_dir = '/tmp/mpf-ipy-profile'
c.Cluster.controller_launcher_class = 'ssh'
c.SSHControllerLauncher.location = '{controller_node}'
c.SSHControllerLauncher.controller_cmd = ['{python_path}', '-m', 'ipyparallel.controller', '--ip={controller_ip}', '--ports={controller_ports}']
c.SSHLauncher.remote_python = '{python_path}'"""
    .format(n=len(engines), engines=repr({e: 1 for e in engines}), controller_node=controller_node, controller_ip=controller_ip, python_path=cluster['global']['python_path'],
            controller_ports=controller_ports))

    with open(os.path.join(profile_dir, 'ipcontroller_config.py'), 'a') as config:
        config.write(f'c.IPController.ports = {repr(controller_ports)}')

    magics_profile_filename = os.path.join(profile_dir, 'startup', '00-mpf_magics.ipy')
    shutil.copy(os.path.join(os.path.dirname(__file__), '00-mpf_magics.ipy'), magics_profile_filename)

    for e in engines:
        p = subprocess.run(['rsync', '--mkpath', magics_profile_filename, f'{e}:/tmp/mpf-ipy-profile/startup/00-mpf_magics.ipy'])
        assert p.returncode == 0

    return controller_node

def add_variable(name: str, values):
    """ Adds the given variable and values to explore in the experiment. """
    assert setup_done, f"mpf.setup must be called first"
    assert name not in variables, f"variable {name} already exists"
    assert name not in roles and name not in links, f"{name} already exists among variables, roles and links"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    if type(values) is range:
        values = list(values)
    variables[name] = Variable(name, values)

def add_wsp_variable(name: str, values=None, range=None):
    """ Adds the given variable and delegates the choice of its values the WSP space filling algorithm.
        To provide a fixed set of values that will be explored, use the values argument.
        To provide a range from which values will be sampled by WSP, use the range argument.
    """
    assert setup_done, f"mpf.setup must be called first"
    assert name not in variables, f"variable {name} already exists"
    assert name not in roles and name not in links, f"{name} already exists among variables, roles and links"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    assert values is not None or range is not None, "One of values and range must be not None"
    variables[name] = WSPVariable(name, values=values or [], range=range or [])

def register_globals(**kwargs):
    """ Updates the experiment global variables with the given names and values. """
    experiment_globals.update(kwargs)

def run(role: Optional[str]=None, link: Optional[str]=None, delay: int=0, parallel=False):
    """ Registers the given function to be executed by a role at given time as part of the experiment. """
    assert setup_done, f"mpf.setup must be called first"
    assert role or link, "at least one of role or link must be set"
    assert role is None or role in roles or role in variables, f"role {role} is not defined in the cluster"
    assert link is None or link in links or link in variables, f"link {link} is not defined in the cluster"
    assert not (role and link) or (((itf := links[link].forward) or (itf := links[link].backward)) and itf.role == role), f"link {link} has no interface belonging to role {role}"
    assert not parallel or delay == 0, "delay can't be used with parallel"
    def inner(func):
        assert 'mpf_ctx' in inspect.getfullargspec(func).args, "functions registered via @mpf.run must accept the 'mpf_ctx' argument"
        functions.append((role, link, delay, parallel, func))
        return func
    return inner

def init(role: str='main'):
    """ Registers the given function to be executed before the start of the experiment. """
    assert setup_done, f"mpf.setup must be called first"
    assert role in roles, f"role {role} is not defined in the cluster"
    def inner(func):
        assert 'mpf_ctx' in inspect.getfullargspec(func).args, "functions registered via @mpf.init must accept the 'mpf_ctx' argument"
        init_functions.append((role, func))
        return func
    return inner

def helper():
    """ Registers the given function as an helper to be pushed on engines. """
    def inner(func):
        helpers.append(func)
        return func
    return inner

def send(role: str, content: dict):
    """ Send files to the specified role
        @param[in]  content: A dictionary whose keys are paths on the host and values the file
                             content to dump
    """
    assert setup_done, f"mpf.setup must be called first"
    assert role in roles, f'role {role} not defined in roles'
    assert type(content) is dict, f'content must be a dict'
    machine_id = roles[role].machine_id
    _apply_send = """
    for dst, content in _files_to_dump.items():
        with open(dst, 'w') as fd:
            fd.write(content)
    del _files_to_dump
    """
    client[machine_id].push({'_files_to_dump': content}, block=True)
    client[machine_id].execute(_apply_send)

def exec_func(role: str, interface: Optional[LinkInterface], function, experiment_values=None, delay=0, ex_ctx={}, parallel=False):
    machine_id = roles[role].machine_id
    mpf_log_global_name = f'_{experiment_id}_{role.encode().hex()}_{repr(interface).encode().hex()}_{function.__name__}_log'
    func_globals = {f.__name__: f for f in helpers}
    func_globals[mpf_log_global_name] = []
    client[machine_id].push(func_globals)
    sleep(delay)
    mpf_ctx = {'roles': {r: {'interfaces': [asdict(itf) for itf in roles[r].interfaces]} for r in roles}, 'role': role}
    if interface:
        mpf_ctx['interface'] = asdict(interface)
    mpf_ctx['mpf_ex_ctx'] = {'namespace': roles[role].namespace, 'cpu_id': roles[role].cpu_id, 'role': role, 'fun': function.__name__, **ex_ctx}
    mpf_ctx['mpf_log'] = mpf_log_global_name
    function_args = inspect.getfullargspec(function).args
    call_args = {arg_name: experiment_values[arg_name] for arg_name in function_args if arg_name not in RESERVED_VARIABLES}
    if 'mpf_ctx' in function_args:
        call_args['mpf_ctx'] = mpf_ctx
    client[machine_id].push(experiment_globals)
    if parallel:
        return client[machine_id].apply_async(function, **call_args), (role, function.__name__, mpf_log_global_name)
    result = client[machine_id].apply_sync(function, **call_args)
    if result is None:
        result = {}
    assert type(result) is dict, "return value of @mpf.run functions should be a dict with the results names and values or None"
    mpf_log = client[machine_id].pull(mpf_log_global_name, block=True)
    for line, out in mpf_log:
        run_logger.info('\n'.join([line] + out), extra={'function': function.__name__, 'role': role})
    client[machine_id].execute(f"del {mpf_log_global_name}")
    return result

def run_experiment(n_runs=3, wsp_target=None, partial_df=None, experiment_id=experiment_id, yield_partial_results=False, log_ex=False) -> Iterator[pd.DataFrame]:
    """ Runs the experiment and yields the results gathered. """
    assert setup_done, f"mpf.setup must be called first"
    if log_ex:
        run_logger.setLevel(logging.INFO)
    if wsp_target is not None:
        wsp_dimensions = sum([type(v) is WSPVariable for v in variables.values()])
        ps = wsp.PointSet.from_random(wsp_target * 10 * wsp_dimensions, wsp_dimensions, 'mpf')
        ps.adaptive_wsp(wsp_target)
        wsp_points.extend(ps.get_remaining())
        del ps
    for role, function in init_functions:
        exec_func(role, None, function, ex_ctx={'exp_id': experiment_id, 'run': 'init'})
    experiments = list(Variable.explore(list(variables.values()))) * n_runs
    assert experiments, "There must be at least one experiment run"
    random.shuffle(experiments)
    results = []
    variable_values = []
    variable_names = list(experiments[0].keys())
    partial_df_idx = 0
    for run_id, experiment_values in enumerate(tqdm(experiments)):
        variable_values.append(tuple(experiment_values.values()))
        if partial_df is not None and partial_df_idx < len(partial_df):
            r = partial_df.iloc[partial_df_idx].to_dict()
            if all(k in r and r[k] == v for k, v in experiment_values.items()):
                partial_df_idx += 1
                results.append({k: v for k, v in r.items() if k not in experiment_values})
                continue
        row = {}
        async_results: List[Tuple[ipp.AsyncResult, Tuple[str, str, str]]] = []  # Stores pending parallel async results with associated role, function name and global mpf_log name
        for role, link, delay, parallel, function in functions:
            if not parallel and async_results:
                ipp.AsyncResult.join([r for r, _ in async_results])
                for result, async_role, async_fn, log_name in [(r.get(), ro, fn, ln) for r, (ro, fn, ln) in async_results]:
                    machine_id = roles[async_role].machine_id
                    if result is None:
                        result = {}
                    assert type(result) is dict, "return value of @mpf.run functions should be a dict with the results names and values or None"
                    assert all(k not in row for k in result.keys()), f"function {async_fn} returned a result name that conflicts with experiment value"
                    row.update(result)
                    mpf_log = client[machine_id].pull(log_name, block=True)
                    for line, out in mpf_log:
                        run_logger.info('\n'.join([line] + out), extra={'function': async_fn, 'role': async_role})
                    client[machine_id].execute(f"del {log_name}")
                async_results = []
            if role in variables:
                role = experiment_values[role]
                assert role in roles, f"Variable role gave value {role} which does not exist"
            if link in variables:
                link = experiment_values[link]
                assert link in links, f"Variable link gave value {link} which does not exist"
            for interface in links[link] if link else [None]:
                r = interface.role if interface and not role else role
                result = exec_func(r, interface, function, experiment_values, delay, ex_ctx={'exp_id': experiment_id, 'run': run_id}, parallel=parallel) # type: ignore
                if parallel:
                    async_results.append(result) # type: ignore
                else:
                    assert all(k not in row for k in result.keys()), f"function {function} returned a result name that conflicts with experiment value"
                    row.update(result)
        assert not async_results, "Some parallel functions were not joined when the experiment run completed"
        results.append(row)
        client.abort()
        for e in engines:
            subprocess.run(['rsync', '-C', '-r', '--mkpath', '--remove-source-files', f"{e}:/dev/shm/mpf_experiments/{experiment_id}/run_{run_id:03}/", f"{experiment_dir}/run_{run_id:03}/"])
        if yield_partial_results:
            yield pd.DataFrame(results, index=pd.MultiIndex.from_tuples(variable_values, names=variable_names))
    if not yield_partial_results:
        yield pd.DataFrame(results, index=pd.MultiIndex.from_tuples(variable_values, names=variable_names))

def start_cluster_and_connect_client(cluster_file: FileIO):
    cluster_profile = f"profile_{os.path.basename(cluster_file.name)}" # type: ignore
    cluster_definition = yaml.safe_load(cluster_file)
    for machine_id, machine_spec in enumerate(cluster_definition['machines']):
        assert 'namespaces' in machine_spec or 'role' in machine_spec, f"machine {machine_spec['hostname']} has no namespaces nor a role"
        machine_roles = []
        if 'role' in machine_spec:
            # TODO: remove useless copy of namespaces if present
            machine_roles.append(machine_spec)
        if 'namespaces' in machine_spec:
            machine_roles.extend(machine_spec['namespaces'])
        for mr in machine_roles:
            assert mr['role'] not in roles, f"role {mr['role']} already exists"
            assert mr['role'] not in variables and mr['role'] not in links, f"{mr['role']} already exists among variables, roles and links"
            interfaces = []
            for itf in mr['interfaces']:
                link_name = itf.get('link')
                assert link_name is None or (link_name not in variables and link_name not in roles), f"{link_name} already exists among variables and roles"
                direction = itf.get('direction', 'forward' if not link_name in links else 'backward')
                interface = LinkInterface(name=itf['name'], ip=itf.get('ip'), role=mr['role'], link=link_name, direction=direction, neighbour=itf.get('neighbour'))
                if link_name and link_name not in links:
                    links[link_name] = Link(link_name, None, None)
                if link := links.get(link_name):
                    assert getattr(link, direction) is None, f"{itf}: A link interface already exists in this direction"
                    setattr(link, direction, interface)
                    if link.forward and link.backward:
                        if link.forward.neighbour is None:  #type: ignore
                            link.forward.neighbour = link.backward.role #type: ignore
                        if link.backward.neighbour is None: #type: ignore
                            link.backward.neighbour = link.forward.role #type: ignore
                interfaces.append(interface)
            roles[mr['role']] = Role(name=mr['role'], machine_id=machine_id, functions=[], interfaces=interfaces, namespace=mr.get('namespace'), cpu_id=mr.get('cpu_id'))
        engines.append(machine_spec['hostname'] if 'user' not in machine_spec else f"{machine_spec['user']}@{machine_spec['hostname']}")

    controller_node = f"{cluster_definition['controller']['user']}@{cluster_definition['controller']['hostname']}"
    try:
        cluster = ipp.Cluster.from_file(profile_dir=cluster_profile)
    except FileNotFoundError:
        create_profile(cluster_profile, cluster_definition)
        subprocess.run(['ipcluster', 'start', f'--profile-dir={cluster_profile}', '--daemonize=True', '--log-level=ERROR'])
        cluster = ipp.Cluster.from_file(profile_dir=cluster_profile)
    return cluster.connect_client_sync(sshserver=controller_node)

def setup(cluster: FileIO):
    global client, experiment_dir, setup_done
    experiment_dir = os.path.join('mpf_experiments', experiment_id)
    os.makedirs(experiment_dir)
    shutil.copy(cluster.name, experiment_dir) # type: ignore
    shutil.copy(sys.argv[0], experiment_dir)

    client = start_cluster_and_connect_client(cluster)
    client.wait_for_engines(timeout=10, block=True)
    cluster.close()
    setup_done = True

def default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='mpf experiment')
    parser.add_argument('-c', '--cluster', metavar='cluster.yaml', type=argparse.FileType('r'), required=True, help='The YAML file describing the cluster')
    return parser

def default_setup():
    parser = default_arg_parser()
    args = parser.parse_args()
    setup(args.cluster)

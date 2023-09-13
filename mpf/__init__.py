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
from tqdm.auto import tqdm


from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
from time import sleep

RESERVED_VARIABLES = {'mpf_ctx'}
random.seed('mpf')

variables: Dict[str, 'Variable'] = {}
functions: List[Tuple[str, int, Callable]] = []
helpers: List[Callable] = []
roles: Dict[str, 'Role'] = {}
experiment_globals: Dict[str, Any] = {}

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

        >>> v1 = Variable('a', [1, 2, 3])
        >>> v2 = Variable('b', ['x', 'y'])
        >>> list(Variable.explore([v1, v2]))
        [{'a': 1, 'b': 'x'}, {'a': 1, 'b': 'y'}, {'a': 2, 'b': 'x'}, {'a': 2, 'b': 'y'}, {'a': 3, 'b': 'x'}, {'a': 3, 'b': 'y'}]
        """
        if len(variables) > 0:
            variable = variables[0]
            for value in variable:
                if len(variables) > 1:
                    for v in Variable.explore(variables[1:]):
                        yield {variable.name: value} | v
                else:
                    yield {variable.name: value}

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
    return controller_node

def add_variable(name: str, values):
    """ Adds the given variable and values to explore in the experiment. """
    assert name not in variables, f"variable {name} already exists"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    if type(values) is range:
        values = list(values)
    variables[name] = Variable(name, values)

def register_globals(**kwargs):
    """ Updates the experiment global variables with the given names and values. """
    experiment_globals.update(kwargs)

def run(role: str='main', delay: int=0):
    """ Registers the given function to be executed by a role at given time. """
    assert role in roles, f"role {role} is not defined in the cluster"
    def inner(func):
        functions.append((role, delay, func))
        return func
    return inner

def helper():
    """ Registers the given function as an helper to be pushed on engines. """
    def inner(func):
        helpers.append(func)
        return func
    return inner

def run_experiment(n_runs=3):
    """ Runs the experiment and returns the results gathered. """
    results = []
    variable_values = []
    experiments = list(Variable.explore(list(variables.values()))) * n_runs
    random.shuffle(experiments)
    for experiment_values in tqdm(experiments):
        row = {}
        for role, delay, function in functions:
            role_id = list(roles).index(role)
            client[role_id].push({f.__name__: f for f in helpers}, block=True)
            sleep(delay)
            mpf_ctx = {'roles': {r: {'interfaces': roles[r].interfaces} for r in roles}}
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

parser = argparse.ArgumentParser(description='mpf experiment')
parser.add_argument('-c', '--cluster', metavar='cluster.yaml', type=argparse.FileType('r'), required=True,help='The YAML file describing the cluster')
args = parser.parse_args()

if args.cluster:
    client = start_cluster_and_connect_client(args.cluster)
    client.wait_for_engines(timeout=30, block=True)

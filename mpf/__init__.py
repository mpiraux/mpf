from io import FileIO
import os
import inspect
import yaml
import subprocess
import argparse
import shutil
import signal
import ipyparallel as ipp


from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from time import sleep

RESERVED_VARIABLES = {'mpf_ctx'}

variables: Dict[str, 'Variable'] = {}
roles: Dict[str, 'Role'] = {}

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
    functions: List[Tuple[int, bool, Any]]  # A list of tuple of delay value, daemonize and IPython function to execute
    interfaces: List[Tuple[str, str]]  # A list of tuple interface name, ip address

def create_profile(profile_dir: str, cluster_file: FileIO):
    """ Populates the given directory with a blank IPython profile structure, enables SSH 
        and sets up the IPython engines based on the given YAML cluster file.
        Creates the roles available from the cluster.
        Returns the [user]@hostname value to use to reach the controller.
    """
    cluster = yaml.safe_load(cluster_file)
    engines = {}
    for machine_spec in cluster['machines']:
        engine = machine_spec['hostname']
        if 'user' in machine_spec:
            engine = f"{machine_spec['user']}@{engine}"
        engines[engine] = 1  # Starts one engine per host
        assert machine_spec['role'] not in roles, f"role {machine_spec['role']} already exists"
        roles[machine_spec['role']] = Role(machine_spec['role'], [], machine_spec['interfaces'])

    # The first machine is also the IPython controller
    controller_node = cluster['machines'][0]['hostname']
    controller_ip = cluster['machines'][0]['interfaces'][0]['ip']

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
    return controller_node

def add_variable(name: str, values):
    """ Adds the given variable and values to explore in the experiment. """
    assert name not in variables, f"variable {name} already exists"
    assert name not in RESERVED_VARIABLES, f"variable {name} is reserved"
    if type(values) is range:
        values = list(values)
    variables[name] = Variable(name, values)

def run(role: str='main', delay: int=0, daemon=False):
    """ Registers the given function to be executed by a role at given time. """
    assert role in roles, f"role {role} is not defined in the cluster"
    r = roles[role]
    def inner(func):
        r.functions.append((delay, daemon, func))
        roles[role] = r
        return func
    return inner

def run_experiment():
    """ Runs the experiment and stops the cluster. """
    for experiment_values in Variable.explore(list(variables.values())):
        for role_id, role in enumerate(roles):
            for delay, daemon, function in roles[role].functions:
                sleep(delay)
                mpf_ctx = {'roles': {r: {'interfaces': roles[r].interfaces} for r in roles}}
                function_args = inspect.getargspec(function).args
                call_args = {arg_name: experiment_values[arg_name] for arg_name in function_args if arg_name not in RESERVED_VARIABLES}
                if 'mpf_ctx' in function_args:
                    call_args['mpf_ctx'] = mpf_ctx
                if daemon:
                    client[role_id].apply(function, **call_args)
                else:
                    print(client[role_id].apply_sync(function, **call_args))
        cluster.signal_engines_sync(signal.SIGINT)
        client.abort()
    subprocess.run(['ipcluster', 'stop', f'--profile-dir={cluster_profile}'])

parser = argparse.ArgumentParser(description='mpf experiment')
parser.add_argument('-c', '--cluster', metavar='cluster.yaml', type=argparse.FileType('r'), required=True,help='The YAML file describing the cluster')
args = parser.parse_args()

if args.cluster:
    cluster_profile = f"profile_{os.path.basename(args.cluster.name)}"
    controller_node = create_profile(cluster_profile, args.cluster)
    subprocess.run(['ipcluster', 'start', f'--profile-dir={cluster_profile}', '--daemonize=True'])
    cluster = ipp.Cluster.from_file(profile_dir=cluster_profile)
    client = cluster.connect_client_sync(sshserver=controller_node)
    client.wait_for_engines()
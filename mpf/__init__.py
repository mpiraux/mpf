from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from time import sleep


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
    """ A role defining what code to run and when to run it. """
    name: str
    functions: List[Tuple[int, Any]]  # A list of tuple of delay value and IPython function to execute

def add_variable(name: str, values):
    """ Adds the given variable and values to explore in the experiment. """
    assert name not in variables, f"variable {name} already exists"
    if type(values) is range:
        values = list(values)
    variables[name] = Variable(name, values)

def run(role: str='main', delay: int=0):
    """ Registers the given function to be executed by a role at given time. """
    r = roles.get(role) or Role(role, [])
    def inner(func):
        r.functions.append((delay, func))
        roles[role] = r
        return func
    return inner

def get_ip_address(role: str, interface: int=0) -> str:
    """ Returns the IP address of the given interface for the given role. """
    return "::1"

def start_experiment():
    """ Runs the experiment. """
    for experiment_values in Variable.explore(list(variables.values())):
        for role in roles:
            for delay, function in roles[role].functions:
                sleep(delay)
                ret = function(**experiment_values)
                if ret:
                    print(experiment_values, list(ret))
# mpf: Minimal Performance Framework

mpf is a tool to write evaluation experiments, exploring variables of interest to understand the performance of computer systems. mpf can compile code, deploy software to a given computer cluster and orchestrate the evaluation to collect data for which graphs can be generated.

## Features

* Experiments are written in IPython.
* Variables can be defined and combined to explore executions scenarios for a given experiment
* Code can be deployed and compiled
* A simple graph is generated from an experiment run

## Defining experiments

A mpf experiment consists of an IPython script. It defines several sections, specifying the variables to explore, the roles taking part in the experiments and the specific code they execute. The following example demonstrates how mpf can run an iperf3 server and client and explore values of the `-P` parallel and `-Z` zerocopy parameter.

```python
#!/usr/bin/env -S ipython --
import mpf

mpf.add_variable('parallel', range(1,9))
mpf.add_variable('zerocopy', {'': 'disabled', '-Z': 'enabled'})

@mpf.run(role='server', daemon=True)
def start_server():
    !iperf3 -s -1 > /dev/null

@mpf.run(role='client', delay=1)
def start_client(mpf_ctx, parallel, zerocopy):
    result = !iperf3 -f k -t 2 -P $parallel $zerocopy -c {mpf_ctx['roles']['server']['interfaces'][0]['ip']} | tail -n 3 | grep -ioE "[0-9.]+ [kmg]bits"
    return {'goodput': result[0]}

mpf.run_experiment()
```

The script defines several parts constituting the experiment. First, it defines two variables that will be explored in the experiment. mpf combines the values of each variable to derive the experiment runs. Second, mpf allows defining functions that will be executed on particular nodes of the cluster. In our example, the function `start_server` will be executed in the background on the machine bearing the `server` role. The `start_client` function will be executed after a one second delay.
This function specifies three parameters. The `mpf_ctx` object describes the context of the experiment, e.g., the IP addresses of each node. `parallel`, `zerocopy` will receive values from the ranges defined above. The `start_client` collect the iperf3 goodput and returns it as part of the result dictionary.

The script can be executed directly and receive a cluster YAML file defining the machines that will run the experiment.

```
$ tests/experiments/iperf.ipy -c tests/experiments/cluster.yaml
```

## Requirements

Running experiments with mpf requires reasonably coherent Python and packages versions between the executor and the remote nodes. `ipyparallel` should be installed on the remote nodes.
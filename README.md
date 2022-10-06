# mpf: Minimal Performance Framework

mpf is a tool to write evaluation experiments, exploring variables of interest to understand the performance of computer systems. mpf can compile code, deploy software to a given computer cluster and orchestrate the evaluation to collect data for which graphs can be generated.

## Features

* Experiments are written in IPython.
* Variables can be defined and combined to explore executions scenarios for a given experiment
* Code can be deployed and compiled
* A simple graph is generated from an experiment run

## Defining experiments

A mpf experiment consists of an IPython script. It defines several sections, specifying the variables to explore, the roles taking part in the experiments and the specific code they execute.

```ipython
import mpf

mpf.add_variable('parallel', range(8)))
mpf.add_variable('zerocopy', {'without': '', 'with': '-Z'})

@mpf.run(as='server')
def start_server(parallel, zerocopy):
    !iperf3 -s

@mpf.run(as='client', delay=1)
def start_client(parallel, zerocopy):
    result = !iperf3 -f k -t 2 -P $parallel $zerocopy -c {mpf.get_ip_address('server', 0)} | tail -n 3 | grep -ioE "[0-9.]+ [kmg]bits"
    mpf.add_result('goodput', result)
```
# ECL-MaxFlow

ECL-MaxFlow is an efficient CUDA implementation of the Push-Relabel algorithm for solving the Maximum Flow problem on GPUs. A full description of the code can be found in our paper (see below).

If you use ECL-MaxFlow, please cite the following publication:

>Avery Vanausdal and Martin Burtscher. "An Efficient Push-Relabel Implementation for Max-Flow Computations on GPUs." Proceedings of the 44th IEEE International Performance Computing and Communications Conference. November 2025. <!--[[doi]]()--> [[abstract]](https://userweb.cs.txstate.edu/~burtscher/abstracts.html#IPCCC25a) [[PDF]](https://userweb.cs.txstate.edu/~burtscher/papers/ipccc25a.pdf)

### Compilation

The code can be compiled as follows:

    make

To target a specific GPU architecture, replace DEVICE_CC in the Makefile with the appropriate [Compute Capability](https://developer.nvidia.com/cuda-gpus).

### Execution

The code takes the following positional arguments:

    ./maxflow <input_graph_path> <source_node> <sink_node> <runs(default=1)>

The source and sink nodes are specified by their index (starting from 0). The optional 'runs' parameter specifies how many times the code should be run. The code reports the median runtime and throughput.

### Input graphs

ECL-Maxflow operates on graphs stored in binary CSR format. Converters to this format and several pre-converted graphs [can be found here](https://cs.txstate.edu/~burtscher/research/ECLgraph/).

The DIMACS graph generators for Acyclic-Dense, Genrmf, Washington, and more can be found [here](http://archive.dimacs.rutgers.edu/pub/netflow/generators/network/). To convert these graphs to CSR format, see the `./convert_dimacs_to_eclgraph/` directory.

To remove nodes outside of the largest connected component and adjust edge weights in the same way as our paper, see the `./preprocess_eclgraph/` directory. This is optional and provided for reproducibility.

*This work has been supported in part by the National Science Foundation under Award #1955367 and by an equipment donation from NVIDIA Corporation.*
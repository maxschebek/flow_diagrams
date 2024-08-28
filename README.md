# Efficient mapping of phase diagrams with conditional Boltzmann Generators
Maximilian Schebek, Michele Invernizzi, Frank Noé, Jutta Rogal 

[![arXiv](https://img.shields.io/badge/arXiv-2406.12378-b31b1b.svg)](https://arxiv.org/abs/2406.12378)
## Abstract
The accurate prediction of phase diagrams is of central importance for both the fundamental understanding of materials as well as for technological applications in material sciences. However, the computational prediction of the relative stability between phases based on their free energy is a daunting task, as traditional free energy estimators require a large amount of simulation data to obtain uncorrelated equilibrium samples over a grid of thermodynamic states. In this work, we develop deep generative machine learning models based on the Boltzmann Generator approach for entire phase diagrams, employing normalizing flows  conditioned on the thermodynamic states, e.g., temperature and pressure, that they map to. By training a single normalizing flow to transform the equilibrium distribution sampled at only one reference thermodynamic state to a wide range of target temperatures and pressures, we can efficiently generate equilibrium samples across the entire phase diagram. Using a permutation-equivariant architecture allows us, thereby, to treat solid and liquid phases on the same footing. We demonstrate our approach by predicting the  solid-liquid coexistence line for a Lennard-Jones system in excellent agreement with state-of-the-art free energy methods while significantly reducing the number of energy evaluations needed.

## Installation
The package and all dependencies (except pytorch) can be installed via
```
python -m pip install -e .
```
This will install a CPU version of JAX - if a GPU is available, it is recommended to remove jax and jaxlib from the setup.py an to install the GPU version following the instruction on the [JAX homepage](https://jax.readthedocs.io/en/latest/installation.html). We used JAX 0.4.23 with python 3.10.

Finally, the repo depends on pytorch (CPU version is sufficient) which may be installed with:
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

```


# Conditional Boltzmann Generators for computing phase diagrams
Maximilian Schebek, Michele Invernizzi, Frank Noé, Jutta Rogal 

## Abstract
The accurate prediction of phase diagrams is of central importance for both the fundamental understanding of materials as well as for technological applications in material sciences. However, the computational prediction of the relative stability between phases based on their free energy is a daunting task, as traditional free energy estimators require a large amount of simulation data to obtain uncorrelated equilibrium samples over a grid of thermodynamic states. In this work, we develop a Boltzmann Generator for entire phase diagrams. We employ normalizing flows conditioned on the thermodynamic states, e.g., temperature and pressure, that they map to. By training a single normalizing flow to map the equilibirum distribution sampled at a single reference thermodynamic state to a wide range of target thermodynamic states, we can efficiently generate equilibrium distributions across the whole phase diagram. Using a permutation-equivariant architecture, we are further able to treat solid and liquid phases on the same footing. We demonstrate our approach by predicting the  solid-liquid coexistence line for a Lennard-Jones system in excellent agreement with state-of-the-art free energy methods while significantly simplifying the computational workflow and reducing computational cost.
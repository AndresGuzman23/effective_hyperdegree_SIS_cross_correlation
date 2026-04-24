# Higher-Order Effective Degree Model

Code and analysis accompanying the paper *"Unveiling the impact of cross-order hyperdegree correlations in contagion processes on hypergraphs"*.

This repository contains the implementation of the higher-order configurational model and the effective hyperdegree SIS model, together with the Gillespie simulations and analysis notebooks used to generate the figures in the paper.

---

## Repository structure

### `library/`

Core Python modules with the main functions used throughout the project, including the configurational model for generating hypergraphs, the effective hyperdegree ODE solver, Gillespie simulation routines, and structural analysis utilities.

There are some varitions of the functions used in different notebooks, but in those cases those varitions are defiend within the notebook. 

### `data/`

Pre-generated hypergraphs and simulation outputs used across the analysis notebooks. This includes the hypergraph realisations and phase diagram data referenced in the results.

### Demonstration notebooks

- **`configurational_model_use.ipynb`** — Demonstrates the higher-order configurational model: how to generate hypergraphs with regular, Poisson, and power-law hyperdegree distributions, and how to introduce tuneable cross-order correlations via the copula method.

- **`effective_hyperdegree_model_use.ipynb`** — Demonstrates the effective hyperdegree SIS model and Gillespie simulations: how to solve the ODE system, run stochastic simulations, and compare the two.

### Analysis folders (by paper section)

Each folder contains the notebooks used to generate the corresponding figures and results:

| Folder | Paper section | Figure |
|--------|--------------|--------|
| `Phase transition/` | Model validation | Figure 1 |
| `Heterogeneity effect (uncorrelated)/` | Section A — Effect of heterogeneity | Figure 3 |
| `Hyperdegree correlation effect/` | Section B — Effect of cross-order correlations | Figure 4 |
| `Hierarchical spread/` | Section C — Hierarchical spreading | Figure 5 |
| `Spreading control/` | Section D — Spreading control | Figure 6 |

---

## Getting started

### Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- NetworkX
- [XGI](https://github.com/xgi-org/xgi) (for hypergraph data structures and visualisation)

Install dependencies with:

```bash
pip install numpy scipy matplotlib networkx xgi
```

### Quick start

1. Start with `configurational_model_use.ipynb` to see how higher-order networks are generated and how cross-order correlations are controlled.
2. Then explore `effective_hyperdegree_model_use.ipynb` to see the SIS dynamics — both the deterministic ODE model and stochastic Gillespie simulations.
3. The analysis folders reproduce the individual figures from the paper.

---

## Citation

If you use this code in your research, please cite:

```
@article{,
  title={Unveiling the impact of cross-order hyperdegree correlations in contagion processes on hypergraphs},
  author={},
  journal={},
  year={}
}
```

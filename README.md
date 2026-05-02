# 🐦 Boids Flocking Simulation

> Implementing Reynolds' *Distributed Behavioral Model* using JAX & JAX-MD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/boids-flocking-simulation/blob/main/boids_flocking_simulation.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-orange)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project is a full JAX-based implementation of the classic **Boids** algorithm by Craig W. Reynolds (SIGGRAPH 1987). It demonstrates how complex, life-like **emergent flocking behaviour** arises purely from three simple local rules — no central controller, no global state.

The simulation is built progressively, layer by layer, using [JAX](https://github.com/google/jax) for JIT compilation and automatic differentiation, and [JAX-MD](https://github.com/google/jax-md) for spatial utilities and molecular-dynamics-style simulation infrastructure.

---

## Features

| Module | Description |
|--------|-------------|
| 🔵 **Free Motion** | Boids initialised randomly in a periodic 2D box |
| ➡️ **Alignment** | Boids steer to match the heading of their neighbours |
| ↔️ **Avoidance** | Soft repulsion prevents collision and pile-up |
| 🔵 **Cohesion** | Boids steer toward the local center of mass |
| 👁️ **Field of View** | Each boid only perceives what is ahead of it |
| 🚧 **Obstacles** | Disk-shaped obstacles the flock navigates around |
| 🦁 **Predators** | Adversarial agents that chase the flock; boids flee |
| ⚡ **Sprint Behaviour** | Predators accelerate in bursts when close to a boid |

---

## Architecture

All rules are expressed as **energy functions**. JAX automatically computes the gradient (force/torque) needed to update each boid's position and orientation:

```
Boids State (R, θ)
        │
        ▼
  Energy Function
  E = E_align + E_avoid + E_cohesion + E_obstacle + E_flee
        │
        ▼ quantity.force(energy_fn)   ← JAX autodiff
  Forces / Torques
        │
        ▼
  Euler Integrator (shift + rotate)
        │
        ▼
  Updated Boids State
```

This formulation means that adding a new rule is as simple as adding a new term to the energy — no manual gradient derivation required.

---

## Quick Start

### Google Colab (Recommended)

Click the badge above, or open the notebook directly — a free GPU runtime is sufficient for 200 boids.

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/boids-flocking-simulation.git
cd boids-flocking-simulation
pip install -r requirements.txt
jupyter notebook boids_flocking_simulation.ipynb
```

> **Note:** JAX with CUDA requires a separate install step. See [JAX installation docs](https://github.com/google/jax#installation).

---

## Simulation Parameters

Key hyperparameters and their effect on emergent behaviour:

```python
# Simulation world
box_size   = 800.0   # Side-length of the periodic box
boid_count = 200     # Number of boids

# Alignment — orient with neighbours
J_align  = 12.0      # Alignment strength
D_align  = 45.0      # Alignment interaction radius

# Avoidance — don't collide
J_avoid  = 25.0      # Repulsion strength
D_avoid  = 30.0      # Collision avoidance radius

# Cohesion — stay together
J_cohesion  = 0.05   # Cohesion strength
D_cohesion  = 40.0   # Cohesion radius

# Field of view
theta_max   = π/3    # ±60° forward cone
```

---

## Notebook Structure

```
boids_flocking_simulation.ipynb
│
├── 1.  Setup & Imports
├── 2.  Boid Representation        ← Boids namedtuple, periodic BC
├── 3.  Free Motion (Baseline)     ← Pure kinematic motion
├── 4.  Alignment Rule             ← Energy + vmap vectorisation
├── 5.  Avoidance Rule             ← Soft repulsion potential
├── 6.  Cohesion Rule              ← Center-of-mass steering
├── 7.  Field of View              ← Perception cone masking
├── 8.  Obstacles                  ← Disk obstacles, avoidance energy
├── 9.  Predators                  ← Two-player energy, chase & flee
├── 10. Internal State             ← Sprint timer, accelerating predator
└── 11. Results & Discussion       ← Summary table + references
```

---

## Technical Highlights

- **JIT compilation** via `@jit` — simulation steps run at near-native speed after the first call.
- **Automatic vectorisation** via `vmap` — pairwise interactions written for one pair, automatically batched over all N² pairs.
- **Autodiff dynamics** via `quantity.force` — forces and torques derived automatically from the energy function.
- **Periodic boundary conditions** via `space.periodic` — boids wrap around the edges of the box seamlessly.

---

## References

- C. W. Reynolds, [*Flocks, Herds, and Schools: A Distributed Behavioral Model*](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.103.7187&rep=rep1&type=pdf), SIGGRAPH 1987
- S. S. Schoenholz & E. D. Cubuk, [*JAX, M.D.: A Framework for Differentiable Physics*](https://arxiv.org/abs/1912.04232), NeurIPS 2020

---

## License

MIT — see [LICENSE](LICENSE) for details.

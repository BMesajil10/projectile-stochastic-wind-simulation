# projectile-stochastic-wind-simulation

Projectile + Wind Simulation (PyBullet)

This repository contains a PyBullet-based projectile trajectory simulator with stochastic wind modeled as:
- Mean drift (per- trajectory trial)
- Ornstein–Uhlenbeck turbulence (time-correlated)
- Smooth gust events (raised-cosine envelope)

The script generates a CSV dataset of 3D trajectories for use in ML training/evaluation.

## Running the simulator creates:
- `Simulation.csv` — trajectory dataset (one row per sampled timestep per trial)

## Requirements
- Python 3.9+ recommended
- Dependencies:
  - `pybullet`
  - `numpy`

## Install
```bash
pip install -r requirements.txt
python simulation_seed1337_trials50.py

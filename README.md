# 2D Godunov / HLLC Hydrodynamics Solver

A compact 2D finite-volume solver for the compressible Euler equations, written from scratch in Python/NumPy.

## Method

- **Riemann solver:** HLLC (Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, ch. 10)
- **Reconstruction:** piecewise-linear (MUSCL-type) with the MinMod slope limiter
- **Time integration:** two-stage predictor–corrector, CFL-limited time step
- **Boundary conditions:** reflecting (rigid) walls in one direction, periodic in the other

## Test problem: Kelvin–Helmholtz instability

The default initial condition follows the Kelvin–Helmholtz setup of Springel (2010): a dense central band (ρ = 2, u = +0.5) between two lighter layers (ρ = 1, u = −0.5) in pressure equilibrium (p = 2.5), seeded with a single-mode sinusoidal velocity perturbation localized at the two interfaces.

## Usage

```bash
pip install numpy matplotlib
python godunov2d.py
```

The run parameters (adiabatic index, box size, resolution, CFL number, end time) are set in the call to `Godunov2(gamma, L, N, c, t_end)` at the bottom of the script.

## Status

Work in progress. Planned: result figures, a resolution-convergence study, comparison with published results, and vectorization of the flux loops.

## References

- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer (2009).
- V. Springel, "E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh", *MNRAS* 401, 791 (2010).

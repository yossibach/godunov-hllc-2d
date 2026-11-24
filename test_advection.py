"""Advection test for the time integration.

A sinusoidal density wave is advected with uniform velocity u0 at constant
pressure. The exact solution is a pure translation, so after time t the wave
must have moved by u0*t. The test measures the actual displacement from the
phase of the first Fourier mode and prints the ratio measured/expected,
which should be 1 for a consistent scheme.
"""
import numpy as np
import godunov2d

N, L = 50, 1.0
U0 = 1.0       # advection velocity
T_END = 0.25   # end time

def IC_advection(L, W, M, N):
    """Density wave along x (the periodic direction)."""
    x, y, dx, dy, X, Y = godunov2d.make_grid(L, W, M, N)
    rho = 1 + 0.2*np.sin(2*np.pi*X/L)
    u = U0*np.ones_like(X)
    v = np.zeros_like(X)
    p = np.ones_like(X)
    return x, y, dx, dy, rho, p, u, v


def wave_phase(rho, x, L):
    """Phase of the first Fourier mode along x (middle row in y)."""
    c = (rho[1:-1, rho.shape[1]//2]*np.exp(-2j*np.pi*x[1:-1]/L)).sum()
    return np.angle(c)

if __name__ == "__main__":
    x, y, ts, us, vs, ps, rhos, cons = godunov2d.Godunov2(1.4, L, L, N, N, 0.3, T_END,
                                                          IC=IC_advection)

    shift = (-(wave_phase(rhos[-1], x, L) - wave_phase(rhos[0], x, L))
             / (2*np.pi)*L) % L
    expected = U0*ts[-1]
    print(f"t = {ts[-1]:.4f}")
    print(f"expected shift = {expected:.4f}, measured shift = {shift:.4f}")
    print(f"ratio = {shift/expected:.3f}   (should be ~1.00)")

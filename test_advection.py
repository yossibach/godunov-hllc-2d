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


def IC_advection(L, W, N, M):
    """Density wave along the periodic direction (array axis 0)."""
    dx, dy = L/N, W/M
    x = np.arange(M)*dx
    y = np.arange(N)*dy
    P = (M-2)*dx                                  # period: interior cells only
    i = np.arange(M)[:, None]*np.ones((1, N))
    rho = 1 + 0.2*np.sin(2*np.pi*i*dx/P)
    u = U0*np.ones((M, N))
    v = np.zeros((M, N))
    p = np.ones((M, N))
    return x, y, dx, dy, rho, p, u, v


def wave_phase(rho, dx, P):
    """Phase of the first Fourier mode along axis 0 (middle column)."""
    n = rho.shape[0]
    xi = np.arange(n)[1:-1]*dx
    c = (rho[1:-1, rho.shape[1]//2]*np.exp(-2j*np.pi*xi/P)).sum()
    return np.angle(c)


if __name__ == "__main__":
    godunov2d.IC = IC_advection      # swap in the advection initial condition
    x, y, ts, us, vs, ps, rhos, cons = godunov2d.Godunov2(1.4, L, N, 0.3, T_END)

    dx = L/N
    P = (N-2)*dx
    shift = (-(wave_phase(rhos[-1], dx, P) - wave_phase(rhos[0], dx, P))
             / (2*np.pi)*P) % P
    expected = U0*ts[-1]
    print(f"t = {ts[-1]:.4f}")
    print(f"expected shift = {expected:.4f}, measured shift = {shift:.4f}")
    print(f"ratio = {shift/expected:.3f}   (should be ~1.00)")

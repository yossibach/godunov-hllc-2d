"""Smoke test for all initial conditions.

Runs every IC for a short time on a square and on a rectangular domain and
checks that the run finishes without NaNs, that mass and energy are conserved
to round-off, and that the circular explosion stays mirror-symmetric.
"""
import numpy as np
import godunov2d as g

T_END = 0.05
CASES = [  # (IC, gamma)
    (g.IC_kh_springel,   5/3),
    (g.IC_kh_sharp,      1.4),
    (g.IC_sod_explosion, 1.4),
]
DOMAINS = [  # (L, W, M, N)
    (1, 1, 20, 20),
    (1, 2, 20, 40),
]

if __name__ == "__main__":
    all_ok = True
    for IC, gamma in CASES:
        for L, W, M, N in DOMAINS:
            x, y, ts, us, vs, ps, rhos, cons = g.Godunov2(gamma, L, W, M, N, 0.3, T_END, IC=IC)
            rho = rhos[-1][1:-1, 1:-1]
            finite = np.isfinite(rho).all()
            drift = np.abs(cons/cons[0] - 1).max()
            ok = finite and drift < 1e-12
            line = f"{IC.__name__:18s} {L}x{W}: finite={finite}, conservation drift={drift:.1e}"
            if IC is g.IC_sod_explosion:
                asym = max(np.abs(rho - rho[::-1, :]).max(), np.abs(rho - rho[:, ::-1]).max())
                ok = ok and asym < 1e-12
                line += f", asymmetry={asym:.1e}"
            print(("PASS  " if ok else "FAIL  ") + line)
            all_ok = all_ok and ok
    print("\nALL PASSED" if all_ok else "\nSOME TESTS FAILED")

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import chq_sav as m

DOMAIN_L = 4.0
GAMMA = 0.01
EPS_Q = 0.001
EPS1 = 0.001         
D0 = 1.0
PHISTAR = 0.4
MEANFIELD = 0.4
IC_AMP = 0.0025
CLAMP_DELTA = 1e-3    # numerical safeguard for A(phi)'s cot(pi*phi); not in CHQ.py
                      # (which only clamps phi<=0), added here for robustness


def _c(p):
    return np.sqrt(1.6 * p ** 2 * (1.0 - p) ** 2 + 0.1)


def _b(p):
    return (1.0 + EPS1) * _c(p) ** 2


def chqpy_coefficients(gamma=GAMMA, W=5.0) -> "m.Coefficients":
   
    A, Ap = m._tanh_A(PHISTAR, 0.5, 10.0)   # identical functional form to Test-2's A
    kappa = lambda p: 1e-3 / (10.0 * p ** 2 + 1e-3)     # NOTE: floor 1e-3, not Test-2's 1e-4
    f = lambda p: W * p ** 2 * (1.0 - p) ** 2
    fp = lambda p: W * (4 * p ** 3 + 2 * p - 6 * p ** 2)   # = W*(4p^3-6p^2+2p)
    return m.Coefficients("chqpy", D0, gamma, EPS_Q, CLAMP_DELTA,
                          _b, _c, A, Ap, kappa, f, fp, eps_b=None)


def make_ic(N, seed=12345):
   
    rng = np.random.default_rng(seed)
    xi = rng.uniform(-IC_AMP, IC_AMP, size=(N, N))
    phi0 = MEANFIELD + xi
    phi0 -= phi0.mean() - MEANFIELD
    q0 = np.zeros((N, N))
    return phi0, q0


if __name__ == "__main__":
    # quick coercivity / sanity printout
    p = np.linspace(0.0, 1.0, 11)
    coef = chqpy_coefficients()
    print(" phi   c(phi)   b(phi)   b-c^2/d0   A(phi)   kappa(phi)   f'(phi)")
    for pv in p:
        cv, bv = _c(pv), _b(pv)
        print(f"{pv:5.2f}  {cv:7.4f}  {bv:7.4f}  {bv-cv**2:9.2e}  "
              f"{float(coef.A(np.array(pv))):7.4f}  {float(coef.kappa(np.array(pv))):10.4e}  "
              f"{float(coef.fprime(np.array(pv))):8.4f}")

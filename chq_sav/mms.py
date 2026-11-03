import numpy as np

LPHI, LQ = 3.0, 2.0          # decay rates of the manufactured modes
APHI, AQ = 0.15, 0.10


def make_mms(grid, coef):
    """Manufactured smooth solution (phi_ex, q_ex) and the forcing that makes
    it an exact solution of the forced CHQ PDE for the given coefficients."""
    kx, ky = 2 * np.pi, 2 * np.pi
    cxcy = np.cos(kx * grid.X) * np.cos(ky * grid.Y)
    sxsy = np.sin(kx * grid.X) * np.sin(ky * grid.Y)

    def phi_ex(t):
        return 0.5 + APHI * np.exp(-LPHI * t) * cxcy

    def q_ex(t):
        return AQ * np.exp(-LQ * t) * sxsy

    def forcing(t):
        p = phi_ex(t)
        q = q_ex(t)
        dpdt = -LPHI * (p - 0.5)
        dqdt = -LQ * q
        mu = -coef.gamma * grid.lap(p) + coef.fprime(p)
        Aq = coef.A(np.clip(p, coef.clamp_delta or 0.0, 1.0 - (coef.clamp_delta or 0.0))) * q \
            if coef.clamp_delta else coef.A(p) * q
        bb = coef.b(p)
        cc = coef.c(p)
        flux_div = (grid.div_wgrad(bb, mu)
                    - grid.dx(cc * grid.dx(Aq)) - grid.dy(cc * grid.dy(Aq)))
        S_phi = dpdt - flux_div
        Aarr = coef.A(p)
        inner = (grid.dx(cc * grid.dx(mu)) + grid.dy(cc * grid.dy(mu))
                 - coef.d0 * grid.lap(Aq))
        q_op = -coef.kappa(p) * q - Aarr * inner + coef.eps_q * grid.lap(q)
        S_q = dqdt - q_op
        return S_phi, S_q

    return phi_ex, q_ex, forcing

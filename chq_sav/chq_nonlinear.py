"""
Independent NONLINEAR reference solver for the CHQ model — BDF2 in time with
f'(φ) treated implicitly by Newton iteration (NO SAV, no auxiliary variable).

Same Fourier–Galerkin spatial discretisation and same coefficient freezing
(φ⋆ = 2φⁿ − φⁿ⁻¹) as the SAV solver `src/chq_sav.py`, so a SAV-vs-nonlinear
comparison isolates exactly one thing: the SAV reformulation of the potential.
If the two agree to O(Δt²) on the same grid, the SAV scheme is faithful to the
model.

Scheme (unknowns φⁿ⁺¹, qⁿ⁺¹; μ eliminated):
    D₂φ = div( b⋆ ∇μ − c⋆ ∇(A⋆ q) ),      μ = −γ Δφ + f'(φⁿ⁺¹)
    D₂q = −κ⋆ q − A⋆ div( c⋆ ∇μ − d0 ∇(A⋆ q) ) + ε_q Δq
Newton linearisation:  f'(φ) ≈ f'(φ_k) + f''(φ_k)(φ − φ_k).
"""
from __future__ import annotations

import numpy as np

from chq_sav import SpectralGrid2D, Coefficients, Frozen, C0_DEFAULT  # reuse


def _fpp(coef, p):
    """f''(φ) by complex-step on f' (coef only carries f, f')."""
    h = 1e-20
    return np.imag(coef.fprime(p + 1j * h)) / h


def _residual(grid, coef, fz, dt, phi, q, phi_n, phi_nm1, q_n, q_nm1, be=False):
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    a = 1.0 / dt if be else 3.0 / (2.0 * dt)
    rhs_p = phi_n / dt if be else (4 * phi_n - phi_nm1) / (2 * dt)
    rhs_q = q_n / dt if be else (4 * q_n - q_nm1) / (2 * dt)
    mu = -g * grid.lap(phi) + coef.fprime(phi)
    F1 = (a * phi - rhs_p
          - (grid.div_wgrad(fz.b, mu)
             - grid.dx(fz.c * grid.dx(fz.A * q)) - grid.dy(fz.c * grid.dy(fz.A * q))))
    div_c_gradmu = grid.dx(fz.c * grid.dx(mu)) + grid.dy(fz.c * grid.dy(mu))
    F2 = (a * q - rhs_q
          + fz.kappa * q
          + fz.A * (div_c_gradmu - d0 * grid.lap(fz.A * q))
          - eps_q * grid.lap(q))
    return F1, F2, mu


def _jac_apply(grid, coef, fz, dt, phik, dphi, dq, be=False):
    """J · (dphi, dq) at the current Newton iterate φ_k."""
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    a = 1.0 / dt if be else 3.0 / (2.0 * dt)
    dmu = -g * grid.lap(dphi) + _fpp(coef, phik) * dphi
    J1 = (a * dphi
          - (grid.div_wgrad(fz.b, dmu)
             - grid.dx(fz.c * grid.dx(fz.A * dq)) - grid.dy(fz.c * grid.dy(fz.A * dq))))
    div_c_grad_dmu = grid.dx(fz.c * grid.dx(dmu)) + grid.dy(fz.c * grid.dy(dmu))
    J2 = (a * dq + fz.kappa * dq
          + fz.A * (div_c_grad_dmu - d0 * grid.lap(fz.A * dq))
          - eps_q * grid.lap(dq))
    return J1, J2


def _assemble_jac(grid, coef, fz, dt, phik, be=False):
    N = grid.ndof
    mm = 2 * N
    J = np.empty((mm, mm))
    e = np.zeros(mm)
    for j in range(mm):
        e[j] = 1.0
        d1, d2 = _jac_apply(grid, coef, fz, dt, phik,
                            e[:N].reshape(grid.shape), e[N:].reshape(grid.shape), be)
        J[:N, j] = d1.ravel()
        J[N:, j] = d2.ravel()
        e[j] = 0.0
    return J


def _jac_precond(grid, coef, fz, dt, phik, be=False):
    """Fourier constant-coefficient preconditioner for the Newton Jacobian —
    same 2×2 (φ̂,q̂) reduction as the SAV solver's, plus the f''(φ_k) shift in the
    μ-row (floored for positivity)."""
    from scipy.sparse.linalg import LinearOperator
    N = grid.ndof
    a = 1.0 / dt if be else 3.0 / (2.0 * dt)
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    bbar = float(np.mean(fz.b))
    cbar = float(np.mean(fz.c))
    Abar = float(np.mean(fz.A))
    Abar2 = float(np.percentile(fz.A**2, 75))
    kbar = float(np.mean(fz.kappa))
    fpp = float(np.mean(_fpp(coef, phik)))
    K2 = -grid._lap_symbol
    K4 = K2**2
    m11 = np.maximum(a + g * bbar * K4 + bbar * fpp * K2, 0.5 * a)  # μ-shift, floored
    m12 = -cbar * Abar * K2
    m21 = -g * Abar * cbar * K4 - Abar * cbar * fpp * K2
    m22 = a + kbar + (d0 * Abar2 + eps_q) * K2
    det = m11 * m22 - m12 * m21
    det = np.where(np.abs(det) < 1e-300, 1.0, det)

    def mv(X):
        X = np.asarray(X, float)
        fp = np.fft.fft2(X[:N].reshape(grid.shape))
        fq = np.fft.fft2(X[N:].reshape(grid.shape))
        Y = np.empty(X.shape, float)
        Y[:N] = np.real(np.fft.ifft2((m22 * fp - m12 * fq) / det)).ravel()
        Y[N:] = np.real(np.fft.ifft2((-m21 * fp + m11 * fq) / det)).ravel()
        return Y

    return LinearOperator((2 * N, 2 * N), matvec=mv)


def nl_bdf2_step(grid, coef, s_nm1, s_n, dt, newton_tol=1e-9, max_newton=25,
                 backend="dense", be=False, gmres_tol=1e-10):
    phi_nm1, q_nm1 = s_nm1
    phi_n, q_n = s_n
    N = grid.ndof
    phi_star = phi_n if be else 2 * phi_n - phi_nm1
    fz = Frozen(grid, coef, phi_star, C0_DEFAULT)
    phi = phi_n.copy()
    q = q_n.copy()
    total_it = 0
    res0 = None
    res_prev = np.inf
    for k in range(max_newton):
        F1, F2, mu = _residual(grid, coef, fz, dt, phi, q, phi_n, phi_nm1, q_n, q_nm1, be)
        res = np.sqrt(grid.nrm2(F1) + grid.nrm2(F2))
        if res0 is None:
            res0 = res
        # converged: small absolute, or shrunk far, or stagnated at the linear-solve floor
        if (res < newton_tol or res < 1e-7 * res0
                or (k > 0 and res > 0.7 * res_prev)):
            break
        res_prev = res
        rhs = -np.concatenate([F1.ravel(), F2.ravel()])
        if backend == "dense":
            J = _assemble_jac(grid, coef, fz, dt, phi, be)
            d = np.linalg.solve(J, rhs)
        else:
            from scipy.sparse.linalg import LinearOperator, gmres
            op = LinearOperator((2 * N, 2 * N), matvec=lambda v: np.concatenate(
                [x.ravel() for x in _jac_apply(grid, coef, fz, dt, phi,
                                               v[:N].reshape(grid.shape),
                                               v[N:].reshape(grid.shape), be)]))
            M = _jac_precond(grid, coef, fz, dt, phi, be)
            it = {"n": 0}
            d, _ = gmres(op, rhs, rtol=gmres_tol, atol=0.0, restart=min(2 * N, 400),
                         maxiter=8, M=M,
                         callback=lambda rk: it.__setitem__("n", it["n"] + 1),
                         callback_type="pr_norm")
            total_it += it["n"]
        phi = phi + d[:N].reshape(grid.shape)
        q = q + d[N:].reshape(grid.shape)
    else:
        raise RuntimeError(f"Newton failed to converge: res={res:.2e}")
    mu = -coef.gamma * grid.lap(phi) + coef.fprime(phi)
    return (phi, q), mu, (k, total_it)     # (newton iters, total GMRES iters)


def nl_run(coef, phi0, q0, T, dt, grid, backend="dense", verbose=False,
           diagnostics=True):
    """BE startup (nonlinear), then BDF2. Returns final (phi, q) and a log
    (each entry: newton = #Newton iters, gmres = total GMRES iters that step).
    diagnostics=False skips the per-step energy for clean timing."""
    s0 = (phi0.copy(), q0.copy())
    s1, _, _ = nl_bdf2_step(grid, coef, s0, s0, dt, backend=backend, be=True)  # BE startup
    s_prev, s_cur = s0, s1
    nsteps = int(round(T / dt))
    log = []
    M0 = grid.integ(phi0)
    for n in range(1, nsteps):
        s_next, mu, (nit, git) = nl_bdf2_step(grid, coef, s_prev, s_cur, dt, backend=backend)
        rec = dict(step=n + 1, t=(n + 1) * dt, newton=nit, gmres=git)
        if diagnostics:
            rec["E"] = (0.5 * coef.gamma * grid.gnrm2(s_next[0])
                        + grid.integ(coef.f(s_next[0])) + 0.5 * grid.nrm2(s_next[1]))
            rec["mass_err"] = abs(grid.integ(s_next[0]) - M0) / (abs(M0) + 1e-300)
            rec["phi_min"] = float(s_next[0].min()); rec["phi_max"] = float(s_next[0].max())
        log.append(rec)
        if verbose and n % max(1, nsteps // 10) == 0:
            print(f"  [nl] step {n+1} newton={nit} gmres={git}")
        s_prev, s_cur = s_cur, s_next
    return s_cur, log

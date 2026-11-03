"""
Minimal periodic Fourier–Galerkin solver for the SAV-BDF2 scheme (S1)–(S4)
for the CHQ viscoelastic phase-separation model.

Derivation and the discrete energy identity (★): analysis/derivation_notes.md.
Design notes for this file:
  * Fourier–Galerkin in space (NO 2/3 dealiasing) so the exact discrete
    integration-by-parts the theorem needs actually holds.
  * The FULL coupled linear system (S1)–(S4) is solved each step (μ eliminated
    analytically; unknowns (φ, q, r)). Two solve backends:
      - "dense": assemble the operator by applying it to unit vectors, then
        numpy.linalg.solve. Ground truth; use for N up to ~64.
      - "gmres": matrix-free GMRES with a Fourier-diagonal preconditioner built
        from mean-frozen coefficients. Use for larger N.

  * NO R-SAV. The scheme is (S1)–(S4) exactly as derived.
  * The discrete energy identity (★) is checked every timestep by default.

Coefficients are frozen at the common extrapolated state φ⋆ = 2φⁿ − φⁿ⁻¹ (clamped
only for A, A', κ — a numerical safeguard, logged).
"""
from __future__ import annotations

import dataclasses
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.sparse.linalg import LinearOperator, gmres

C0_DEFAULT = 1.0


# ======================================================================
# Spectral grid
# ======================================================================
class SpectralGrid2D:
    """Periodic [0,Lx) x [0,Ly), Nx x Ny collocation, FFT spectral derivatives."""

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float):
        self.Nx, self.Ny, self.Lx, self.Ly = Nx, Ny, Lx, Ly
        self.shape = (Nx, Ny)
        self.ndof = Nx * Ny
        x = Lx * np.arange(Nx) / Nx
        y = Ly * np.arange(Ny) / Ny
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        kx = 2 * np.pi / Lx * fftfreq(Nx, d=1.0 / Nx)
        ky = 2 * np.pi / Ly * fftfreq(Ny, d=1.0 / Ny)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        # zero the Nyquist mode for odd-order derivatives (even N only)
        self._dxmask = np.ones_like(self.KX)
        self._dymask = np.ones_like(self.KY)
        if Nx % 2 == 0:
            self._dxmask[Nx // 2, :] = 0.0
        if Ny % 2 == 0:
            self._dymask[:, Ny // 2] = 0.0
        self.K2 = self.KX**2 + self.KY**2
        # Laplacian symbol MUST be dx∘dx + dy∘dy (not −|k|²) so that the exact
        # discrete identity ⟨Δu,v⟩ = −⟨∇u,∇v⟩ holds — matters for the (★) identity
        # once the solution has Nyquist-mode content (even N).
        self._lap_symbol = -(self.KX**2 * self._dxmask + self.KY**2 * self._dymask)
        self.cell = (Lx / Nx) * (Ly / Ny)
        self._Dx_mat = self._Dy_mat = self._Lap_mat = None

    # --- dense spectral operators (lazy; for the 'dense' solve backend) ---
    def _dense_1d(self, N, L, mask_even):
        k = 2 * np.pi / L * fftfreq(N, d=1.0 / N)
        if mask_even and N % 2 == 0:
            k = k.copy()
            k[N // 2] = 0.0
        col = np.fft.fft(np.eye(N), axis=0)
        return np.real(np.fft.ifft(1j * k[:, None] * col, axis=0))

    @property
    def Dx_mat(self):
        if self._Dx_mat is None:
            D1 = self._dense_1d(self.Nx, self.Lx, True)
            self._Dx_mat = np.kron(D1, np.eye(self.Ny))
        return self._Dx_mat

    @property
    def Dy_mat(self):
        if self._Dy_mat is None:
            D1 = self._dense_1d(self.Ny, self.Ly, True)
            self._Dy_mat = np.kron(np.eye(self.Nx), D1)
        return self._Dy_mat

    @property
    def Lap_mat(self):
        if self._Lap_mat is None:
            self._Lap_mat = self.Dx_mat @ self.Dx_mat + self.Dy_mat @ self.Dy_mat
        return self._Lap_mat

    # --- derivatives (real field -> real field) ---
    def dx(self, u):
        return np.real(ifft2(1j * self.KX * self._dxmask * fft2(u)))

    def dy(self, u):
        return np.real(ifft2(1j * self.KY * self._dymask * fft2(u)))

    def lap(self, u):
        # = dx(dx(u)) + dy(dy(u)) exactly (see _lap_symbol note in __init__)
        return np.real(ifft2(self._lap_symbol * fft2(u)))

    def div_wgrad(self, w, u):
        """div( w grad u ) with w a field."""
        return self.dx(w * self.dx(u)) + self.dy(w * self.dy(u))

    # --- inner products / norms (L2 over Omega) ---
    def integ(self, u):
        return float(np.sum(u) * self.cell)

    def ip(self, u, v):
        return float(np.sum(u * v) * self.cell)

    def nrm2(self, u):
        return self.ip(u, u)

    def gnrm2(self, u):
        return self.ip(self.dx(u), self.dx(u)) + self.ip(self.dy(u), self.dy(u))


# ======================================================================
# Coefficient sets
# ======================================================================
@dataclasses.dataclass
class Coefficients:
    name: str
    d0: float
    gamma: float
    eps_q: float
    clamp_delta: float
    b: callable
    c: callable
    A: callable
    Aprime: callable
    kappa: callable
    f: callable
    fprime: callable
    # optional exact margin (min over [0,1] of b - c^2/d0); informational
    eps_b: float | None = None


def _tanh_A(phistar, amp, slope):
    cotps = 1.0 / np.tan(np.pi * phistar)

    def A(p):
        return amp * (1.0 + np.tanh(slope * (cotps - 1.0 / np.tan(np.pi * p))))

    def Aprime(p):
        g = slope * (cotps - 1.0 / np.tan(np.pi * p))
        dg = slope * np.pi / np.sin(np.pi * p) ** 2
        return amp * (1.0 - np.tanh(g) ** 2) * dg

    return A, Aprime


def benchmark_test2(gamma=1e-3) -> Coefficients:
    d0, eps_q, eps_b = 1.0, 1e-3, 1e-3
    c = lambda p: p * (1.0 - p) / np.sqrt(10.0)
    b = lambda p: c(p) ** 2 / d0 + eps_b
    A, Ap = _tanh_A(0.4, 0.5, 10.0)
    kappa = lambda p: 1e-3 / (10.0 * p**2 + 1e-4)
    f = lambda p: (p - 0.95) ** 2 * (p - 0.05) ** 2
    fp = lambda p: 2 * (p - 0.95) * (p - 0.05) ** 2 + 2 * (p - 0.95) ** 2 * (p - 0.05)
    return Coefficients("benchmark_test2", d0, gamma, eps_q, 1e-3, b, c, A, Ap, kappa, f, fp, eps_b)


def benchmark_test1(gamma=1e-3) -> Coefficients:
    d0, eps_q, eps_b = 1.0, 1e-3, 1e-3
    c = lambda p: (4.0 / np.sqrt(10.0)) * p * (1.0 - p)
    b = lambda p: c(p) ** 2 / d0 + eps_b
    A, Ap = _tanh_A(0.5, 5e-3, 5.0)
    kappa = lambda p: 1e-2 / (10.0 * p**2 + 1e-4)
    f = lambda p: (p - 0.95) ** 2 * (p - 0.05) ** 2
    fp = lambda p: 2 * (p - 0.95) * (p - 0.05) ** 2 + 2 * (p - 0.95) ** 2 * (p - 0.05)
    return Coefficients("benchmark_test1", d0, gamma, eps_q, 1e-3, b, c, A, Ap, kappa, f, fp, eps_b)


def without_coupling(coef: Coefficients) -> Coefficients:
    zero = lambda p: np.zeros_like(np.asarray(p, dtype=float))
    return dataclasses.replace(coef, name=coef.name + "_noA", A=zero, Aprime=zero)


def generic() -> Coefficients:
    d0, gamma, eps_q, eps_b = 1.2, 0.3, 0.15, 0.2
    c = lambda p: 0.4 + 0.2 * np.cos(p)
    b = lambda p: c(p) ** 2 / d0 + eps_b
    A = lambda p: 0.5 + 0.3 * np.sin(p)
    Ap = lambda p: 0.3 * np.cos(p)
    kappa = lambda p: 0.6 + 0.3 * p**2
    f = lambda p: (p - 0.3) ** 2 * (p - 0.7) ** 2        # smooth wells at 0.3, 0.7
    fp = lambda p: 2 * (p - 0.3) * (p - 0.7) ** 2 + 2 * (p - 0.3) ** 2 * (p - 0.7)
    return Coefficients("generic", d0, gamma, eps_q, None, b, c, A, Ap, kappa, f, fp, eps_b)


# ======================================================================
# Frozen coefficients at φ⋆
# ======================================================================
class Frozen:
    """b⋆, c⋆, A⋆, κ⋆, G⋆ for a given extrapolated φ⋆ and shift constant C0."""

    def __init__(self, grid: SpectralGrid2D, coef: Coefficients, phi_star, C0: float):
        d = coef.clamp_delta
        if d is None:
            pcl = phi_star
            self.clamp_frac = 0.0
        else:
            pcl = np.clip(phi_star, d, 1.0 - d)
            self.clamp_frac = float(np.mean((phi_star < d) | (phi_star > 1.0 - d)))
        self.b = coef.b(pcl)
        self.c = coef.c(pcl)
        self.A = coef.A(pcl)
        self.kappa = coef.kappa(pcl)
        self.d0 = coef.d0
        self.min_b_minus_c2d0 = float(np.min(self.b - self.c**2 / coef.d0))
        Ef_star = grid.integ(coef.f(phi_star)) + C0
        assert Ef_star > 0, f"E_f(phi*)+C0 = {Ef_star} <= 0 : increase C0"
        self.G = coef.fprime(phi_star) / np.sqrt(Ef_star)   # G⋆ uses unclamped φ⋆


# ======================================================================
# One BDF2 step — the linear operator on (φ, q, r) with μ eliminated
# ======================================================================
def _bdf2_apply(grid, coef, fz, dt, X):
    """X = (phi, q, r) flattened. Returns L(X) flattened.
    μ = −γΔφ + r G⋆ has been substituted into (S1),(S3)."""
    N = grid.ndof
    phi = X[:N].reshape(grid.shape)
    q = X[N:2 * N].reshape(grid.shape)
    r = X[2 * N]
    a = 3.0 / (2.0 * dt)
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q

    # μ = −γ Δφ + r G⋆   (its ∇ enters via div(b⋆∇μ), A⋆ div(c⋆∇μ))
    # div(b⋆ ∇μ) = −γ div(b⋆ ∇Δφ) + r div(b⋆ ∇G⋆)
    div_b_gradmu = -g * grid.div_wgrad(fz.b, grid.lap(phi)) + r * grid.div_wgrad(fz.b, fz.G)
    # A⋆ div(c⋆ ∇μ) = −γ A⋆ div(c⋆ ∇Δφ) + r A⋆ div(c⋆ ∇G⋆)
    A_div_c_gradmu = fz.A * (-g * grid.div_wgrad(fz.c, grid.lap(phi))
                             + r * grid.div_wgrad(fz.c, fz.G))

    # (S1):  a φ − div(b⋆∇μ) + div(c⋆∇(A⋆ q)) = rhs_phi
    row_phi = a * phi - div_b_gradmu + grid.div_wgrad(fz.c, fz.A * q)

    # (S3):  a q + κ⋆ q + A⋆ div(c⋆∇μ) − d0 A⋆ Δ(A⋆ q) − ε_q Δq = rhs_q
    row_q = (a * q + fz.kappa * q + A_div_c_gradmu
             - d0 * fz.A * grid.lap(fz.A * q) - eps_q * grid.lap(q))

    # (S4):  a r − (a/2) ⟨G⋆, φ⟩ = rhs_r
    row_r = a * r - 0.5 * a * grid.ip(fz.G, phi)

    out = np.empty_like(X)
    out[:N] = row_phi.ravel()
    out[N:2 * N] = row_q.ravel()
    out[2 * N] = row_r
    return out


def _bdf2_rhs(grid, coef, fz, dt, phi_n, phi_nm1, q_n, q_nm1, r_n, r_nm1,
             S_phi=None, S_q=None):
    """RHS for one BDF2 step.  Optional forcing S_phi, S_q (arrays at t^{n+1}) for
    manufactured-solution convergence tests.  The forcing enters only (S1) and
    (S3); (S4) `D₂r = ½⟨G⋆, D₂φ⟩` uses difference quotients of the unknowns and
    is untouched."""
    N = grid.ndof
    tdt = 2.0 * dt
    rhs = np.empty(2 * N + 1)
    rhs[:N] = ((4 * phi_n - phi_nm1) / tdt).ravel()
    rhs[N:2 * N] = ((4 * q_n - q_nm1) / tdt).ravel()
    rhs[2 * N] = (4 * r_n - r_nm1) / tdt - 0.5 * grid.ip(fz.G, 4 * phi_n - phi_nm1) / tdt
    if S_phi is not None:
        rhs[:N] += S_phi.ravel()
    if S_q is not None:
        rhs[N:2 * N] += S_q.ravel()
    return rhs


def _fourier_preconditioner(grid, coef, fz, dt, be=False):
    """M^{-1} from a *constant*-coefficient reduction of the frozen operator:
    solves the 2×2 (φ̂, q̂) system per Fourier mode exactly (incl. the φ↔q
    coupling), drops only the rank-1 r-coupling.

    Coefficient choices (from experiments/wp5_precond_bench.py):
      b  -> mean(b⋆)            near mesh-independent when b⋆ varies mildly
                                (Test-1 variable mobility: ~5–36 GMRES iters)
      A² -> 75th pct of A⋆²     helps the q-block diffusion when A⋆ sweeps 0→1
                                (Test-2 sharp A(φ)) — still grows ~N^0.5 there;
                                a multigrid q-block preconditioner is the real
                                fix (deferred, WP6)."""
    N = grid.ndof
    a = 1.0 / dt if be else 3.0 / (2.0 * dt)
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    bbar = float(np.mean(fz.b))
    cbar = float(np.mean(fz.c))
    Abar = float(np.mean(fz.A))
    Abar2 = float(np.percentile(fz.A**2, 75))
    kbar = float(np.mean(fz.kappa))
    K2 = -grid._lap_symbol           # consistent with the operator's Laplacian
    K4 = K2**2

    # symbols of the constant-coefficient block operator (see _assemble_dense):
    m11 = a + g * bbar * K4                       # row_phi · φ̂
    m12 = -cbar * Abar * K2                       # row_phi · q̂  (div(c∇(Aq)))
    m21 = -g * Abar * cbar * K4                   # row_q · φ̂    (A div(c∇μ̃))
    m22 = a + kbar + (d0 * Abar2 + eps_q) * K2    # row_q · q̂
    det = m11 * m22 - m12 * m21
    det = np.where(np.abs(det) < 1e-300, 1.0, det)

    r_scale = 1.0 if be else a          # r-row coefficient (see _assemble_dense)

    def mv(X):
        fphi = fft2(X[:N].reshape(grid.shape))
        fq = fft2(X[N:2 * N].reshape(grid.shape))
        pphi = (m22 * fphi - m12 * fq) / det
        pq = (-m21 * fphi + m11 * fq) / det
        out = np.empty_like(X)
        out[:N] = np.real(ifft2(pphi)).ravel()
        out[N:2 * N] = np.real(ifft2(pq)).ravel()
        out[2 * N] = X[2 * N] / r_scale
        return out

    return LinearOperator((2 * N + 1, 2 * N + 1), matvec=mv)


def _assemble_dense(grid, coef, fz, dt, be=False):
    """Build the (2N²+1)x(2N²+1) matrix for (S1),(S3),(S4) with μ eliminated,
    directly from dense spectral operators.  be=True → first-order backward-Euler
    startup (a = 1/Δt, and the r-row is  r − ½⟨G⋆,φ⟩)."""
    N = grid.ndof
    a = 1.0 / dt if be else 3.0 / (2.0 * dt)
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    Dx, Dy, Lap = grid.Dx_mat, grid.Dy_mat, grid.Lap_mat
    I = np.eye(N)
    Gv = fz.G.ravel()
    DIVB = Dx @ (fz.b.ravel()[:, None] * Dx) + Dy @ (fz.b.ravel()[:, None] * Dy)
    DIVC = Dx @ (fz.c.ravel()[:, None] * Dx) + Dy @ (fz.c.ravel()[:, None] * Dy)
    Adiag = fz.A.ravel()[:, None]
    Lap_A = Lap * fz.A.ravel()[None, :]        # Lap @ diag(A)

    # row_phi = [ aI + γ DIVB@Lap  |  DIVC@diag(A)  |  −DIVB@Gv ]
    r1_phi = a * I + g * (DIVB @ Lap)
    r1_q = DIVC * fz.A.ravel()[None, :]
    r1_r = -(DIVB @ Gv)

    # row_q = [ −γ diag(A)@DIVC@Lap | aI + diag(κ) − d0 diag(A)@Lap@diag(A) − ε_q Lap | diag(A)@DIVC@Gv ]
    r3_phi = -g * (fz.A.ravel()[:, None] * (DIVC @ Lap))
    r3_q = (a * I + np.diag(fz.kappa.ravel())
            - d0 * (fz.A.ravel()[:, None] * Lap_A)
            - eps_q * Lap)
    r3_r = fz.A.ravel() * (DIVC @ Gv)

    # row_r = [ −(a/2) cell Gv | 0 | a ]
    m = 2 * N + 1
    Amat = np.zeros((m, m))
    Amat[:N, :N] = r1_phi
    Amat[:N, N:2 * N] = r1_q
    Amat[:N, 2 * N] = r1_r
    Amat[N:2 * N, :N] = r3_phi
    Amat[N:2 * N, N:2 * N] = r3_q
    Amat[N:2 * N, 2 * N] = r3_r
    r_scale = 1.0 if be else a
    Amat[2 * N, :N] = -0.5 * r_scale * grid.cell * Gv
    Amat[2 * N, 2 * N] = r_scale
    return Amat


@dataclasses.dataclass
class SolveInfo:
    backend: str
    iters: int = 0
    resid: float = 0.0


def bdf2_step(grid, coef, s_nm1, s_n, dt, C0=C0_DEFAULT, backend="dense",
              gmres_tol=1e-11, gmres_maxiter=12, forcing=None, t_new=None):
    """s = (phi, q, r).  Returns (s_np1, mu_np1, frozen, SolveInfo).
    forcing: optional callable t -> (S_phi, S_q) (manufactured-solution tests)."""
    phi_nm1, q_nm1, r_nm1 = s_nm1
    phi_n, q_n, r_n = s_n
    phi_star = 2.0 * phi_n - phi_nm1
    fz = Frozen(grid, coef, phi_star, C0)
    Sp = Sq = None
    if forcing is not None:
        Sp, Sq = forcing(t_new)
    rhs = _bdf2_rhs(grid, coef, fz, dt, phi_n, phi_nm1, q_n, q_nm1, r_n, r_nm1, Sp, Sq)
    N = grid.ndof

    if backend == "dense":
        Amat = _assemble_dense(grid, coef, fz, dt)
        X = np.linalg.solve(Amat, rhs)
        info = SolveInfo("dense", resid=float(
            np.linalg.norm(Amat @ X - rhs) / (np.linalg.norm(rhs) + 1e-300)))
    elif backend == "gmres":
        op = LinearOperator((2 * N + 1, 2 * N + 1),
                            matvec=lambda v: _bdf2_apply(grid, coef, fz, dt, v))
        M = _fourier_preconditioner(grid, coef, fz, dt)
        it = {"n": 0}
        rst = min(2 * N + 1, 300)
        X, flag = gmres(op, rhs, rtol=gmres_tol, atol=0.0, restart=rst,
                        maxiter=gmres_maxiter, M=M,
                        callback=lambda rk: it.__setitem__("n", it["n"] + 1),
                        callback_type="pr_norm")
        resid = float(np.linalg.norm(op.matvec(X) - rhs) / (np.linalg.norm(rhs) + 1e-300))
        info = SolveInfo("gmres", iters=it["n"], resid=resid)
        if resid > max(1e-8, 1e3 * gmres_tol):
            raise RuntimeError(
                f"GMRES converged poorly (flag={flag}, true rel resid={resid:.2e}, "
                f"iters={it['n']}) — tighten tol / improve preconditioner")
    else:
        raise ValueError(backend)

    phi1 = X[:N].reshape(grid.shape)
    q1 = X[N:2 * N].reshape(grid.shape)
    r1 = float(X[2 * N])
    mu1 = -coef.gamma * grid.lap(phi1) + r1 * fz.G
    return (phi1, q1, r1), mu1, fz, info


# ======================================================================
# Two-solve SAV decoupling  (derivation_notes.md §6.3, Lemma "𝓛 nonsingular")
# ======================================================================
def bdf2_step_twosolve(grid, coef, s_nm1, s_n, dt, C0=C0_DEFAULT,
                       denom_tol=1e-8):
    """One BDF2 step via the SAV two-solve reduction instead of the (2N²+1)
    block solve.  Eliminate rⁿ⁺¹ from (S4) and μⁿ⁺¹ from (S2); the remaining
    (φ,q) system is  𝓛(φ,q) + η p = F  with  η = ½⟨G⋆,φⁿ⁺¹⟩ a scalar and 𝓛 the
    r-free block operator.  Solve  x⁽¹⁾ = 𝓛⁻¹F,  x⁽²⁾ = 𝓛⁻¹(−p)  (two solves,
    same operator), then

        η = ½⟨G⋆,φ⁽¹⁾⟩ / (1 − ½⟨G⋆,φ⁽²⁾⟩),
        (φ,q)ⁿ⁺¹ = x⁽¹⁾ + η x⁽²⁾,   rⁿ⁺¹ = η + ξⁿ.

    Lemma 𝓛 nonsingular ⇒ the denominator ≠ 0; it is checked at runtime and a
    near-zero value raises (caller should fall back to `bdf2_step`).

    Dense 𝓛 assembly — for verification / small N.  Returns the same tuple shape
    as `bdf2_step`: (s_np1, mu_np1, fz, SolveInfo).
    """
    phi_nm1, q_nm1, r_nm1 = s_nm1
    phi_n, q_n, r_n = s_n
    N = grid.ndof
    phi_star = 2.0 * phi_n - phi_nm1
    fz = Frozen(grid, coef, phi_star, C0)

    Amat = _assemble_dense(grid, coef, fz, dt)      # (2N²+1)²
    L = Amat[:2 * N, :2 * N]                        # r-free block operator 𝓛
    p = Amat[:2 * N, 2 * N].copy()                  # the r-coupling column = [p₁; p₃]

    tdt = 2.0 * dt
    Fbase = np.empty(2 * N)
    Fbase[:N] = ((4 * phi_n - phi_nm1) / tdt).ravel()
    Fbase[N:] = ((4 * q_n - q_nm1) / tdt).ravel()
    # ξⁿ = ⅓[ 4rⁿ − rⁿ⁻¹ − ½⟨G⋆, 4φⁿ − φⁿ⁻¹⟩ ]
    xi_n = (4 * r_n - r_nm1
            - 0.5 * grid.ip(fz.G, 4 * phi_n - phi_nm1)) / 3.0

    x1 = np.linalg.solve(L, Fbase - xi_n * p)
    x2 = np.linalg.solve(L, -p)

    phi1, q1 = x1[:N].reshape(grid.shape), x1[N:].reshape(grid.shape)
    phi2, q2 = x2[:N].reshape(grid.shape), x2[N:].reshape(grid.shape)

    denom = 1.0 - 0.5 * grid.ip(fz.G, phi2)
    if abs(denom) < denom_tol:
        raise RuntimeError(
            f"two-solve denominator 1 − ½⟨G⋆,φ⁽²⁾⟩ = {denom:.2e} near zero — "
            f"fall back to bdf2_step")
    eta = 0.5 * grid.ip(fz.G, phi1) / denom

    phi_np1 = phi1 + eta * phi2
    q_np1 = q1 + eta * q2
    r_np1 = float(eta + xi_n)
    mu_np1 = -coef.gamma * grid.lap(phi_np1) + r_np1 * fz.G
    info = SolveInfo("twosolve", iters=2, resid=abs(denom))
    return (phi_np1, q_np1, r_np1), mu_np1, fz, info


# ======================================================================
# First-order SAV / backward-Euler startup step
# ======================================================================
def _be_apply(grid, coef, fz, dt, X):
    N = grid.ndof
    phi = X[:N].reshape(grid.shape)
    q = X[N:2 * N].reshape(grid.shape)
    r = X[2 * N]
    a = 1.0 / dt
    g, d0, eps_q = coef.gamma, coef.d0, coef.eps_q
    div_b_gradmu = -g * grid.div_wgrad(fz.b, grid.lap(phi)) + r * grid.div_wgrad(fz.b, fz.G)
    A_div_c_gradmu = fz.A * (-g * grid.div_wgrad(fz.c, grid.lap(phi))
                             + r * grid.div_wgrad(fz.c, fz.G))
    row_phi = a * phi - div_b_gradmu + grid.div_wgrad(fz.c, fz.A * q)
    row_q = (a * q + fz.kappa * q + A_div_c_gradmu
             - d0 * fz.A * grid.lap(fz.A * q) - eps_q * grid.lap(q))
    row_r = r - 0.5 * grid.ip(fz.G, phi)
    out = np.empty_like(X)
    out[:N] = row_phi.ravel()
    out[N:2 * N] = row_q.ravel()
    out[2 * N] = row_r
    return out


def _be_step(grid, coef, s_from, dt, C0=C0_DEFAULT, backend="dense", gmres_tol=1e-11,
             forcing=None, t_new=None):
    """One first-order SAV / backward-Euler step from state s_from = (φ,q,r).
    Coefficients frozen at s_from[0]. Returns (φ¹, q¹, r¹)."""
    phi_p, q_p, r_p = s_from
    fz = Frozen(grid, coef, phi_p, C0)
    N = grid.ndof
    a = 1.0 / dt
    rhs = np.empty(2 * N + 1)
    rhs[:N] = (a * phi_p).ravel()
    rhs[N:2 * N] = (a * q_p).ravel()
    rhs[2 * N] = r_p - 0.5 * grid.ip(fz.G, phi_p)
    if forcing is not None:
        Sp, Sq = forcing(t_new)
        rhs[:N] += Sp.ravel()
        rhs[N:2 * N] += Sq.ravel()
    if backend == "dense":
        Amat = _assemble_dense(grid, coef, fz, dt, be=True)
        X = np.linalg.solve(Amat, rhs)
    else:
        op = LinearOperator((2 * N + 1, 2 * N + 1),
                            matvec=lambda v: _be_apply(grid, coef, fz, dt, v))
        M = _fourier_preconditioner(grid, coef, fz, dt, be=True)
        X, flag = gmres(op, rhs, rtol=gmres_tol, atol=0.0,
                        restart=min(2 * N + 1, 300), maxiter=12, M=M)
        resid = float(np.linalg.norm(op.matvec(X) - rhs) / (np.linalg.norm(rhs) + 1e-300))
        if resid > max(1e-8, 1e3 * gmres_tol):
            raise RuntimeError(f"startup BE GMRES poor: resid {resid:.2e} (flag {flag})")
    return (X[:N].reshape(grid.shape), X[N:2 * N].reshape(grid.shape), float(X[2 * N]))


def startup(grid, coef, s0, dt, C0=C0_DEFAULT, backend="dense", method="be",
            gmres_tol=1e-12, forcing=None, t0=0.0):
    """Produce s¹ = (φ¹, q¹, r¹) from s⁰.

    method="be"          : one backward-Euler / first-order SAV step
                           (local error O(Δt²) ⇒ O(Δt²) global contribution).
    method="richardson"  : Richardson-extrapolated BE — one BE step of Δt (sa)
                           and two BE steps of Δt/2 (sb); s¹ = 2 sb − sa.
                           Local error O(Δt³): a genuinely 2nd-order-accurate
                           startup, used to check that the "be" startup does not
                           contaminate the global BDF2 rate.
    """
    if method == "be":
        return _be_step(grid, coef, s0, dt, C0, backend, gmres_tol,
                        forcing=forcing, t_new=t0 + dt)
    if method == "richardson":
        sa = _be_step(grid, coef, s0, dt, C0, backend, gmres_tol,
                      forcing=forcing, t_new=t0 + dt)
        sh = _be_step(grid, coef, s0, dt / 2, C0, backend, gmres_tol,
                      forcing=forcing, t_new=t0 + dt / 2)
        sb = _be_step(grid, coef, sh, dt / 2, C0, backend, gmres_tol,
                      forcing=forcing, t_new=t0 + dt)
        return (2 * sb[0] - sa[0], 2 * sb[1] - sa[1], 2.0 * sb[2] - sa[2])
    raise ValueError(f"unknown startup method {method!r}")


# ======================================================================
# Diagnostics
# ======================================================================
def _Phi(grid, u, v, grad=False):
    n2 = grid.gnrm2 if grad else grid.nrm2
    return n2(u) + n2(2 * u - v)


def modified_energy(grid, coef, s_a, s_b, C0=C0_DEFAULT):
    """𝓔 built from levels (a=n+1, b=n)."""
    pa, qa, ra = s_a
    pb, qb, rb = s_b
    return 0.25 * (coef.gamma * _Phi(grid, pa, pb, grad=True)
                   + _Phi(grid, qa, qb)
                   + 2.0 * (ra**2 + (2 * ra - rb) ** 2)) - C0


def dissipation(grid, coef, fz, mu, q):
    d0, eps_q = coef.d0, coef.eps_q
    mux, muy = grid.dx(mu), grid.dy(mu)
    Aq = fz.A * q
    Aqx, Aqy = grid.dx(Aq), grid.dy(Aq)
    t1 = grid.integ((fz.b - fz.c**2 / d0) * (mux**2 + muy**2))
    t2 = grid.integ((fz.c * mux - d0 * Aqx) ** 2 + (fz.c * muy - d0 * Aqy) ** 2) / d0
    t3 = grid.integ(fz.kappa * q**2)
    t4 = eps_q * grid.gnrm2(q)
    return t1 + t2 + t3 + t4


def true_energy(grid, coef, phi, q):
    return (0.5 * coef.gamma * grid.gnrm2(phi)
            + grid.integ(coef.f(phi))
            + 0.5 * grid.nrm2(q))


def star_residual(grid, coef, fz, s_np1, s_n, s_nm1, mu_np1, dt, C0=C0_DEFAULT):
    """Residual of  𝓔ⁿ⁺¹ − 𝓔ⁿ + ¼[γ‖∇D²φ‖² + ‖D²q‖² + 2(D²r)²] + Δt D̃ⁿ⁺¹ = 0."""
    p1, q1, r1 = s_np1
    pn, qn, rn = s_n
    pm, qm, rm = s_nm1
    E_np1 = modified_energy(grid, coef, s_np1, s_n, C0)
    E_n = modified_energy(grid, coef, s_n, s_nm1, C0)
    D2phi = p1 - 2 * pn + pm
    D2q = q1 - 2 * qn + qm
    D2r = r1 - 2 * rn + rm
    d2 = 0.25 * (coef.gamma * grid.gnrm2(D2phi) + grid.nrm2(D2q) + 2 * D2r**2)
    Dt = dissipation(grid, coef, fz, mu_np1, q1)
    res = (E_np1 - E_n) + d2 + dt * Dt
    # scale relative to the ENERGY magnitude (stays O(1)); the term-sum scale
    # collapses to ~0 near steady state and makes a fine absolute residual look huge.
    scale_E = abs(E_np1) + abs(E_n) + C0
    scale_terms = abs(E_np1 - E_n) + d2 + dt * abs(Dt)
    return dict(residual=res,
                rel=abs(res) / (scale_E + 1e-300),        # robust: vs energy magnitude
                rel_terms=abs(res) / (scale_terms + 1e-300),
                dEmod=E_np1 - E_n, Dtilde=Dt, d2terms=d2)


# ======================================================================
# Time-stepping driver
# ======================================================================
@dataclasses.dataclass
class RunResult:
    grid: SpectralGrid2D
    coef: Coefficients
    dt: float
    t: np.ndarray
    phi: list          # snapshots kept (default: last only unless keep_all)
    q: list
    r: list
    log: list          # per-step diagnostics dicts
    final: tuple       # (phi, q, r) at T


def run(coef: Coefficients, phi0, q0, T, dt, grid: SpectralGrid2D,
        C0=C0_DEFAULT, backend="dense", check_star=True, star_rtol=1e-8,
        startup_method="be", gmres_tol=1e-12, keep_every=0, verbose=False,
        forcing=None, diagnostics=True, post_step=None):
    """Integrate (S1)–(S4) from (phi0, q0) to time T.

    check_star: assert  |(★) residual| ≤ star_rtol · (|𝓔ⁿ⁺¹| + |𝓔ⁿ| + C₀)
    (i.e. relative to the energy magnitude, which stays O(1)) every step.
    startup_method: "be" (default) or "richardson" (2nd-order-accurate startup).
    forcing: optional callable t -> (S_phi, S_q); manufactured-solution tests
             (disables the (★) assertion — forcing does work on the energy).
    diagnostics: if False, skip the ~50-FFT per-step diagnostic bundle
                 ((★) residual, energies, dissipation) — for clean timing.
                 Only mass + GMRES iters/resid are logged; check_star is off.
    post_step: optional callable (grid, (phi,q,r)) -> (phi,q,r), applied to the
               solution immediately after every step (including the startup
               step) before logging/snapshotting/(★)-checking. NOT part of the
               proven scheme (S1)-(S4) — e.g. a post-hoc spectral filter for an
               aliasing diagnostic — so (★)/energy behaviour under it is
               empirical, not covered by Theorem 5.1's identity. Default None
               (no-op; all existing verified behaviour is unaffected).
    """
    if forcing is not None:
        check_star = False
    if not diagnostics:
        check_star = False
    r0 = float(np.sqrt(grid.integ(coef.f(phi0)) + C0))
    s0 = (phi0.copy(), q0.copy(), r0)

    s1 = startup(grid, coef, s0, dt, C0, backend=backend, method=startup_method,
                 gmres_tol=gmres_tol, forcing=forcing, t0=0.0)
    if post_step is not None:
        s1 = post_step(grid, s1)
    nsteps = int(round(T / dt))

    log = []
    M0 = grid.integ(phi0)
    snaps_phi, snaps_q, snaps_r, tsnap = [phi0.copy()], [q0.copy()], [r0], [0.0]

    s_prev, s_cur = s0, s1
    t = dt
    for n in range(1, nsteps):
        s_next, mu_next, fz, info = bdf2_step(grid, coef, s_prev, s_cur, dt, C0,
                                             backend=backend, gmres_tol=gmres_tol,
                                             forcing=forcing, t_new=t + dt)
        if post_step is not None:
            s_next = post_step(grid, s_next)
        t += dt

        if not diagnostics:
            log.append(dict(step=n + 1, t=t, mass=grid.integ(s_next[0]),
                            solve_iters=info.iters, solve_resid=info.resid,
                            clamp_frac=fz.clamp_frac))
            if keep_every and (n % keep_every == 0):   # trajectory snapshots are cheap
                snaps_phi.append(s_next[0].copy()); snaps_q.append(s_next[1].copy())
                snaps_r.append(s_next[2]); tsnap.append(t)
            s_prev, s_cur = s_cur, s_next
            continue

        star = star_residual(grid, coef, fz, s_next, s_cur, s_prev, mu_next, dt, C0)
        r_true = float(np.sqrt(grid.integ(coef.f(s_next[0])) + C0))
        rec = dict(
            step=n + 1, t=t,
            mass=grid.integ(s_next[0]),
            mass_err=abs(grid.integ(s_next[0]) - M0) / (abs(M0) + 1e-300),
            Emod=modified_energy(grid, coef, s_next, s_cur, C0),
            dEmod=star["dEmod"],
            E_true=true_energy(grid, coef, s_next[0], s_next[1]),
            star_rel=star["rel"], star_rel_terms=star["rel_terms"],
            star_abs=star["residual"], Dtilde=star["Dtilde"],
            sav_err=abs(s_next[2] - r_true),
            min_b_c2d0=fz.min_b_minus_c2d0,
            clamp_frac=fz.clamp_frac,
            solve_iters=info.iters, solve_resid=info.resid,
            phi_min=float(s_next[0].min()), phi_max=float(s_next[0].max()),
        )
        log.append(rec)
        if verbose and (n % max(1, nsteps // 20) == 0 or n < 3):
            print(f"  step {rec['step']:5d} t={t:8.4f}  Emod={rec['Emod']:.6e} "
                  f"dEmod={rec['dEmod']:+.2e}  star_rel={rec['star_rel']:.1e}  "
                  f"mass_err={rec['mass_err']:.1e}  clamp={rec['clamp_frac']:.2f}  "
                  f"min(b-c2/d0)={rec['min_b_c2d0']:.2e}  its={rec['solve_iters']}")

        if check_star:                       # both checks off under forcing
            if star["rel"] > star_rtol:
                raise AssertionError(
                    f"(★) residual too large at step {n+1}: rel={star['rel']:.3e} "
                    f"(abs={star['residual']:.3e}); backend={backend}")
            if star["dEmod"] > 1e-9 * (abs(rec["Emod"]) + 1.0):
                raise AssertionError(
                    f"modified energy increased at step {n+1}: dEmod={star['dEmod']:.3e}")

        if keep_every and (n % keep_every == 0):
            snaps_phi.append(s_next[0].copy()); snaps_q.append(s_next[1].copy())
            snaps_r.append(s_next[2]); tsnap.append(t)

        s_prev, s_cur = s_cur, s_next

    snaps_phi.append(s_cur[0].copy()); snaps_q.append(s_cur[1].copy())
    snaps_r.append(s_cur[2]); tsnap.append(t)
    return RunResult(grid, coef, dt, np.array(tsnap), snaps_phi, snaps_q, snaps_r,
                     log, s_cur)

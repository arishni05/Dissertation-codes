import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.interpolate import RegularGridInterpolator

# Heston parameters
K = 100.0  # strike
r = 0.02   # risk-free rate
q = 0.05   # dividend yield
T = 0.15   # maturity
sigma = 0.3    # vol of variance
rho = -0.9   # correlation
kappa = 1.5    # mean reversion
theta = 0.04   # long-run variance

# Domain
Smin, Smax = 0.0, 2.0 * K  # 0...200
vmin, vmax = 0.0, 0.5      # 0...0.5

# Evaluation point
S0 = 101.52
v0 = 0.05412

#Explicit scheme: U^n = U^{n+1} + dt * L(U^{n+1})
def heston_fdm(N_S=80, N_v=40, N_t=3000):
    # Grids
    S = np.linspace(Smin, Smax, N_S)
    v = np.linspace(vmin, vmax, N_v)
    dS, dv = S[1] - S[0], v[1] - v[0]

    # dt from target N_t, then CFL for stability
    dt_user = T / (N_t - 1)

    # CFL-based time step estimates (for explicit diffusion stability)
    dt_S = 0.45 * (dS**2) / ((vmax + 1e-12) * (Smax**2))
    dt_v = 0.45 * (dv**2) / ((sigma**2) * (vmax + 1e-12))
    dt = min(dt_user, dt_S, dt_v)

    # Recompute N_t and dt consistently
    N_t = int(np.ceil(T / dt)) + 1
    dt = T / (N_t - 1)

    print(f"dS={dS:.4f}, dv={dv:.5f}, dt={dt:.7f}, N_t={N_t}")

    # Solution array: U(S_i, v_j, t_n)
    U = np.zeros((N_S, N_v, N_t))

    # Terminal condition at t = T (European call payoff)
    U[:, :, -1] = np.maximum(S[:, None] - K, 0.0)

    # Time stepping: go backward n = N_t-2, ..., 0
    for n in reversed(range(N_t - 1)):
        t_n = n * dt
        tau = T - t_n  # time-to-maturity for boundary conditions

        # Interior points
        for i in range(1, N_S - 1):
            for j in range(1, N_v - 1):
                dU_dS = (U[i + 1, j, n + 1] - U[i - 1, j, n + 1]) / (2.0 * dS)
                d2U_dS2 = (U[i + 1, j, n + 1] - 2.0 * U[i, j, n + 1] + U[i - 1, j, n + 1]) / (dS**2)

                dU_dv = (U[i, j + 1, n + 1] - U[i, j - 1, n + 1]) / (2.0 * dv)
                d2U_dv2 = (U[i, j + 1, n + 1] - 2.0 * U[i, j, n + 1] + U[i, j - 1, n + 1]) / (dv**2)

                d2U_dSdv = (U[i + 1, j + 1, n + 1] - U[i + 1, j - 1, n + 1]
                    - U[i - 1, j + 1, n + 1] + U[i - 1, j - 1, n + 1]
                ) / (4.0 * dS * dv)

                S_i = S[i]
                v_j = v[j]

                # Heston operator L(U) at (i,j,n+1) with dividend yield q
                L = (0.5 * v_j * S_i**2 * d2U_dS2
                    + rho * sigma * v_j * S_i * d2U_dSdv
                    + 0.5 * sigma**2 * v_j * d2U_dv2
                    + (r - q) * S_i * dU_dS         
                    + kappa * (theta - v_j) * dU_dv
                    - r * U[i, j, n + 1]
                )

                U[i, j, n] = U[i, j, n + 1] + dt * L

        # Boundary conditions in S
        U[0, :, n] = 0.0  # S = 0 → call ~ 0

        # Large-S asymptotic with dividends:
        # U(S, v, tau) ~ S*e^{-q*tau} - K*e^{-r*tau} as S -> infinity
        U[-1, :, n] = Smax * np.exp(-q * tau) - K * np.exp(-r * tau)

        # Simple Neumann boundaries in v
        U[:, 0, n]  = U[:, 1, n]      # dv derivative ~ 0 at v=0
        U[:, -1, n] = U[:, -2, n]     # dv derivative ~ 0 at v=vmax

        U[:, :, n] = np.nan_to_num(U[:, :, n])

    return S, v, U

# Run solver and print price at (S0, v0)
start = time()
S_fdm, v_fdm, U_fdm = heston_fdm()
runtime = time() - start
print(f"FDM runtime: {runtime:.3f}s (stored {U_fdm.shape[2]} time slices)")

# t = 0 slice (current option value)
U_t0 = U_fdm[:, :, 0]

# Interpolator 
interp_fdm = RegularGridInterpolator(
    (S_fdm, v_fdm),
    U_t0,
    bounds_error=False,
    fill_value=None,
)

FDPrice = interp_fdm([S0, v0])[()]  
print("Explicit FDM price (bilinear) at (S0, v0):", FDPrice)

# 3D surface at t=0
S_mesh, v_mesh = np.meshgrid(S_fdm, v_fdm, indexing="xy")
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S_mesh, v_mesh, U_t0.T, cmap="plasma", linewidth=0, antialiased=True)
ax.set_title("Explicit FDM Heston surface (t = 0)")
ax.set_xlabel("S")
ax.set_ylabel("v")
ax.set_zlabel("U")
plt.tight_layout()
plt.show()

# Heatmap of U(S,v) at t=0
plt.figure(figsize=(6, 5))
plt.title("Heatmap of U(S,v) at t = 0")
plt.imshow(
    U_t0.T,
    extent=[S_fdm[0], S_fdm[-1], v_fdm[0], v_fdm[-1]],
    origin='lower',
    aspect='auto',
    cmap='plasma'
)
plt.colorbar(label="U")
plt.xlabel("S")
plt.ylabel("v")
plt.tight_layout()
plt.show()

# Alternate 3D view 
def plot_surface(Sm, Vm, U2d, title):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sm, Vm, U2d.T, cmap="plasma", linewidth=0, antialiased=True)
    ax.set_xlabel('S')
    ax.set_ylabel('v')
    ax.set_zlabel('U')
    ax.set_title(title)
    ax.view_init(elev=30, azim=225)  
    plt.tight_layout()
    plt.show()

plot_surface(S_mesh, v_mesh, U_t0, 'Heston Explicit FDM surface at t = 0')

# S-slices at fixed v values
v_levels = [0.01, 0.10, 0.30, 0.45] 
plt.figure(figsize=(7, 5))
for v0_plot in v_levels:
    j = np.argmin(np.abs(v_fdm - v0_plot))
    plt.plot(S_fdm, U_t0[:, j], label=f"v={v_fdm[j]:.2f}")
plt.axvline(K, color='k', linestyle='--', alpha=0.6)
plt.title("Slices: U vs S at fixed v (t = 0)")
plt.xlabel("S")
plt.ylabel("U")
plt.legend()
plt.tight_layout()
plt.show()

# v-slices at fixed S values
S_levels = [50, 100, 150]
plt.figure(figsize=(7, 5))
for S0_plot in S_levels:
    i = np.argmin(np.abs(S_fdm - S0_plot))
    plt.plot(v_fdm, U_t0[i, :], label=f"S={S_fdm[i]:.0f}")
plt.title("Slices: U vs v at fixed S (t = 0)")
plt.xlabel("v")
plt.ylabel("U")
plt.legend()
plt.tight_layout()
plt.show()

# Time evolution at S = K for different v
N_t = U_fdm.shape[2]
dt = T / (N_t - 1)
t_grid = np.linspace(0.0, T, N_t)
tau_vals = T - t_grid  

iK = np.argmin(np.abs(S_fdm - K))

plt.figure(figsize=(7, 5))
for v0_plot in v_levels:
    j = np.argmin(np.abs(v_fdm - v0_plot))
    # U[iK,j,n]: t=0 is n=0, t=T is n=N_t-1, but tau is reversed
    plt.plot(tau_vals, U_fdm[iK, j, :], label=f"v={v_fdm[j]:.2f}")
plt.title("Time Evolution at S ≈ K")
plt.xlabel("τ (time-to-maturity)")
plt.ylabel("U")
plt.legend()
plt.tight_layout()
plt.show()

# Greeks (Delta, Gamma vs S; Vega vs v) at t=0
dS = S_fdm[1] - S_fdm[0]
dv = v_fdm[1] - v_fdm[0]

def central_diff(y, h):
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2*h)
    dy[0] = (y[1] - y[0]) / h
    dy[-1] = (y[-1] - y[-2]) / h
    return dy

def second_diff(y, h):
    d2 = np.zeros_like(y)
    d2[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / h**2
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2

# Delta & Gamma vs S 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for v0_plot in v_levels:
    j = np.argmin(np.abs(v_fdm - v0_plot))
    delta = central_diff(U_t0[:, j], dS)
    plt.plot(S_fdm, delta, label=f"v={v_fdm[j]:.2f}")
plt.axvline(K, ls="--", color="k")
plt.title("Delta vs S at t = 0")
plt.xlabel("S")
plt.ylabel("Delta")
plt.legend()

plt.subplot(1, 2, 2)
for v0_plot in v_levels:
    j = np.argmin(np.abs(v_fdm - v0_plot))
    gamma = second_diff(U_t0[:, j], dS)
    plt.plot(S_fdm, gamma, label=f"v={v_fdm[j]:.2f}")
plt.axvline(K, ls="--", color="k")
plt.title("Gamma vs S at t = 0")
plt.xlabel("S")
plt.ylabel("Gamma")
plt.legend()

plt.tight_layout()
plt.show()

# Vega vs v 
plt.figure(figsize=(7, 5))
for S0_plot in S_levels:
    i = np.argmin(np.abs(S_fdm - S0_plot))
    vega = central_diff(U_t0[i, :], dv)
    plt.plot(v_fdm, vega, label=f"S={S_fdm[i]:.0f}")
plt.title("Vega vs v at t = 0")
plt.xlabel("v")
plt.ylabel("Vega")
plt.legend()
plt.tight_layout()
plt.show()
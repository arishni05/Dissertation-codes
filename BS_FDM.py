import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from scipy.special import erf   

# Black–Scholes parameters
r = 0.05          # risk-free interest rate
sigma = 0.2       # volatility
K = 1.0           # strike
T = 1.0           # maturity
S_max = 4.0 * K   

# Analytic BS formula 
def N_cdf(x):
    """Standard normal CDF using scipy.erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def bs_call_analytic(S, t):
    """
    Analytic Black-Scholes call price V(S,t)
    Works for BOTH scalars and numpy arrays.
    """
    S = np.asarray(S)
    tau = T - t  # time to maturity

    if tau <= 0:
        return np.maximum(S - K, 0.0)

    S_safe = np.maximum(S, 1e-16)

    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    V = S_safe * N_cdf(d1) - K * np.exp(-r * tau) * N_cdf(d2)

    return np.where(S <= 0, 0.0, V)

# Explicit FTCS solver in tau = T - t
def bs_explicit_1d(N_S=201, N_t=20000):
    S = np.linspace(0, S_max, N_S)
    dS = S[1] - S[0]

    dt_user = T / (N_t - 1)

    # Stability limit approximations
    diff_denom = sigma**2 * S_max**2 + 1e-16
    conv_denom = r * S_max + 1e-16

    dt_diff = 0.45 * dS**2 / diff_denom
    dt_conv = 0.45 * dS / conv_denom
    dt_reac = 0.9 / r

    dt = min(dt_user, dt_diff, dt_conv, dt_reac)

    N_t_eff = int(np.ceil(T / dt)) + 1
    dt = T / (N_t_eff - 1)

    tau = np.linspace(0, T, N_t_eff)

    # Initial condition at tau=0 (t = T)
    U = np.zeros((N_S, N_t_eff))
    U[:, 0] = np.maximum(S - K, 0)

    # Boundary at S=0
    U[0, :] = 0.0

    # Explicit stepping in tau (forward)
    for n in range(N_t_eff - 1):
        for i in range(1, N_S - 1):
            Si = S[i]
            d2U = (U[i+1, n] - 2*U[i, n] + U[i-1, n]) / dS**2
            dU  = (U[i+1, n] - U[i-1, n]) / (2*dS)

            U[i, n+1] = U[i, n] + dt * (
                0.5 * sigma**2 * Si**2 * d2U
                + r * Si * dU
                - r * U[i, n]
            )

        # Boundary at S = Smax
        U[-1, n+1] = S_max - K * np.exp(-r * tau[n+1])

    return S, tau, U, dS, dt

# Run FDM
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
})

start = time()
S, tau, U_tau, dS, dt_tau = bs_explicit_1d(N_S=201, N_t=20000)
print(f"Runtime: {time()-start:.3f}s, dS={dS:.4e}, dt_tau={dt_tau:.4e}")

# Convert back to actual time t in [0, T]
# tau = 0 -> t = T, tau = T -> t = 0
t = T - tau[::-1]
V_num = U_tau[:, ::-1]

# Analytic reference on same (S, t) grid
V_exact = np.array([bs_call_analytic(S, ti) for ti in t]).T
Error = V_num - V_exact

# Accuracy check at a single point
S0 = 1.3
t0 = 0.4

# Nearest grid
iS = np.argmin(np.abs(S - S0))
it = np.argmin(np.abs(t - t0))

V_fdm_point = float(V_num[iS, it])
V_exact_point = float(bs_call_analytic(S0, t0))

print("\n--- Pointwise Check ---")
print(f"Requested point:   (S0={S0}, t0={t0})")
print(f"Nearest grid point (Sg={S[iS]:.5f}, tg={t[it]:.5f})")
print(f"FDM   price = {V_fdm_point:.8f}")
print(f"Exact price = {V_exact_point:.8f}")
print(f"Abs error   = {abs(V_fdm_point - V_exact_point):.3e}")

# Global errors on full grid
AbsErr = np.abs(Error)
L2 = float(np.sqrt(np.mean(AbsErr**2)))
Linf = float(np.max(AbsErr))

print("\n--- Global Error Metrics ---")
print(f"L2   error = {L2:.3e}")
print(f"Linf error = {Linf:.3e}")

# Time slices
plt.figure(figsize=(15,10))
for frac in (0.0, 0.1, 0.3, 0.6, 1.0):
    idx = int(frac * (len(t) - 1))
    plt.plot(S, V_num[:, idx], label=f"FDM t={t[idx]:.2f}")
    plt.plot(S, V_exact[:, idx], 'k--', alpha=0.8, label=f"Exact t={t[idx]:.2f}")
plt.title("Black–Scholes FDM vs Analytic")
plt.xlabel("S"); plt.ylabel("V(S,t)")
plt.legend(ncol=2); plt.grid(True)
plt.tight_layout(); plt.show()

# Error at t = 0  (current time)
plt.figure(figsize=(15,10))
plt.plot(S, np.abs(V_num[:, 0] - V_exact[:, 0]))
plt.title("Absolute Error at t = 0")
plt.xlabel("S"); plt.ylabel("|Error|")
plt.grid(True)
plt.tight_layout(); plt.show()

# 3D surface of V(S,t) 
Tm, Sm = np.meshgrid(t, S, indexing='xy')
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Tm, Sm, V_num, cmap=cm.viridis)
ax.set_title("FDM Solution V(S,t)")
ax.set_xlabel("t"); ax.set_ylabel("S"); ax.set_zlabel("V")
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# 3D error surface
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Tm, Sm, AbsErr, cmap=cm.inferno)
ax.set_title("Absolute Error |FDM - Exact|")
ax.set_xlabel("t"); ax.set_ylabel("S"); ax.set_zlabel("Error")
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# Heatmaps
plt.figure(figsize=(15,12))
plt.imshow(V_num, extent=[t[0], t[-1], S[0], S[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label="V(S,t)")
plt.xlabel("t"); plt.ylabel("S")
plt.title("Heatmap of FDM Solution V(S,t)")
plt.tight_layout(); plt.show()

plt.figure(figsize=(15,12))
plt.imshow(AbsErr, extent=[t[0], t[-1], S[0], S[-1]],
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="|Error|")
plt.xlabel("t"); plt.ylabel("S")
plt.title("Heatmap of Absolute Error")
plt.tight_layout(); plt.show()
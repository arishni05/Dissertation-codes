# Explicit FTCS solver for the 1D heat equation:
#   u_t = D * u_xx,  x in (0,1), t in (0,1]
#   IC: u(x,0) = sin(pi x)
#   Dirichlet BCs: u(0,t) = u(1,t) = 0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time

# Problem parameters
D = 0.1        # diffusion coefficient 
L = 1.0        # spatial domain [0,1]
T = 1.0        # final time

def heat_fdm_1d(N_x=201, N_t=10_000):
    assert N_x >= 3
    x = np.linspace(0.0, L, N_x)
    dx = x[1] - x[0]

    # stable dt (CFL condition)
    dt_user = T / max(N_t - 1, 1)
    dt_cfl  = 0.45 * dx * dx / (D + 1e-16)
    dt = min(dt_user, dt_cfl)

    N_t_eff = int(np.ceil(T / dt)) + 1
    dt = T / (N_t_eff - 1)
    t = np.linspace(0.0, T, N_t_eff)

    # Initial condition
    U0 = np.sin(np.pi * x)

    U = np.zeros((N_x, N_t_eff))
    U[:, 0] = U0

    # Dirichlet BCs
    U[0, 0]  = 0.0
    U[-1, 0] = 0.0

    coef = D * dt / (dx * dx)

    for n in range(N_t_eff - 1):
        d2 = U[2:, n] - 2.0 * U[1:-1, n] + U[:-2, n]
        U[1:-1, n + 1] = U[1:-1, n] + coef * d2
        U[0,   n + 1] = 0.0
        U[-1,  n + 1] = 0.0

    return x, t, U, dx, dt

plt.rcParams.update({
    'font.size': 16,         
    'axes.titlesize': 20,    
    'axes.labelsize': 18,    
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# Run simulation
start = time()
x, t, U, dx, dt = heat_fdm_1d(N_x=201, N_t=10_000)
print(f"Runtime: {time()-start:.3f}s, dx={dx:.4e}, dt={dt:.4e}")

# Analytic solution and error
def u_exact(x, tval):
    return np.exp(- (np.pi**2) * D * tval) * np.sin(np.pi * x)

U_exact = np.array([u_exact(x, ti) for ti in t]).T
Error = U - U_exact

# Time-slice comparisons
plt.figure(figsize=(15,10))
for frac in (0.0, 0.1, 0.3, 0.6, 1.0):
    idx = int(frac * (len(t) - 1))
    plt.plot(x, U[:, idx], label=f"FDM t={t[idx]:.2f}")
    plt.plot(x, U_exact[:, idx], 'k--', alpha=0.8, label=f"Exact t={t[idx]:.2f}")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.title("1D Heat Equation (FTCS vs Analytic)")
plt.legend(ncol=2); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(15,10))
plt.plot(x, np.abs(U[:, -1] - U_exact[:, -1]))
plt.xlabel("x"); plt.ylabel("|Error|")
plt.title("Pointwise absolute error at final time (t=1)")
plt.grid(True); plt.tight_layout(); plt.show()

# 3D Surface u(x,t)
Tm, Xm = np.meshgrid(t, x, indexing='xy')  # Tm: shape (N_x, N_t), Xm: (N_x, N_t)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Tm, Xm, U, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('x'); ax.set_zlabel('u')
ax.set_title('Numerical solution u(x,t) (FDM)')
ax.view_init(elev=30, azim=225) 
plt.tight_layout(); plt.show()

# 3D Error surface
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Tm, Xm, np.abs(Error), cmap=cm.inferno, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('x'); ax.set_zlabel('|Error|')
ax.set_title('Pointwise absolute error |U - U_exact|')
ax.view_init(elev=30, azim=225)  
plt.tight_layout(); plt.show()

# Heatmaps
plt.figure(figsize=(15,12))
plt.imshow(U, extent=[t[0], t[-1], x[0], x[-1]], aspect='auto',
           origin='lower', cmap='viridis')
plt.colorbar(label='u(x,t)'); plt.xlabel('t'); plt.ylabel('x')
plt.title('Heatmap of u(x,t) (FDM)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(15,12))
plt.imshow(np.abs(Error), extent=[t[0], t[-1], x[0], x[-1]], aspect='auto',
           origin='lower', cmap='inferno')
plt.colorbar(label='|Error|'); plt.xlabel('t'); plt.ylabel('x')
plt.title('Heatmap of |U - U_exact|')
plt.tight_layout(); plt.show()
# Explicit FTCS solver for the 2D heat equation:
#   u_t = D * (u_xx + u_yy),  (x,y) in (0,1)x(0,1), t in (0,1]
#   u(x,y,0) = sin(pi x) sin(pi y)
#   u = 0 on boundary of the square

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time

# Problem parameters
D = 0.1        
Lx, Ly = 1.0, 1.0
T = 1.0       


def heat_fdm_2d(Nx=81, Ny=81, Nt=5000):
    assert Nx >= 3 and Ny >= 3

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Stable dt (CFL condition for 2D)
    # Sufficient stability: dt <= 1 / [2D(1/dx^2 + 1/dy^2)]
    dt_user = T / max(Nt - 1, 1)
    dt_cfl = 0.45 * 1.0 / (2.0 * D * (1.0/dx**2 + 1.0/dy**2))
    dt = min(dt_user, dt_cfl)

    Nt_eff = int(np.ceil(T / dt)) + 1
    dt = T / (Nt_eff - 1)
    t = np.linspace(0.0, T, Nt_eff)

    # Initial condition: u(x,y,0) = sin(pi x) sin(pi y)
    X, Y = np.meshgrid(x, y, indexing='xy')  # shape (Ny, Nx)
    U = np.zeros((Nt_eff, Ny, Nx))          # time-major: U[n,j,i]
    U0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    U[0, :, :] = U0

    # Dirichlet BCs at t=0
    U[0, 0, :] = 0.0
    U[0, -1, :] = 0.0
    U[0, :, 0] = 0.0
    U[0, :, -1] = 0.0

    lamx = D * dt / dx**2
    lamy = D * dt / dy**2

    # Time stepping 
    for n in range(Nt_eff - 1):
        Un = U[n]
        Un_next = U[n + 1]

        # Update interior points (j=1...Ny-2, i=1...Nx-2)
        Un_next[1:-1, 1:-1] = (
            Un[1:-1, 1:-1]
            + lamx * (Un[1:-1, 2:] - 2.0 * Un[1:-1, 1:-1] + Un[1:-1, :-2])
            + lamy * (Un[2:, 1:-1] - 2.0 * Un[1:-1, 1:-1] + Un[:-2, 1:-1])
        )

        # Dirichlet boundary = 0
        Un_next[0, :] = 0.0
        Un_next[-1, :] = 0.0
        Un_next[:, 0] = 0.0
        Un_next[:, -1] = 0.0

    return x, y, t, U, dx, dy, dt


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
x, y, t, U, dx, dy, dt = heat_fdm_2d(Nx=81, Ny=81, Nt=5000)
print(f"Runtime: {time()-start:.3f}s, dx={dx:.4e}, dy={dy:.4e}, dt={dt:.4e}")


# Analytic solution u(x,y,t) = exp(-2*pi^2*D*t) sin(pi x) sin(pi y)
def u_exact(x, y, tval):
    X, Y = np.meshgrid(x, y, indexing='xy')
    return np.exp(-2.0 * (np.pi ** 2) * D * tval) * np.sin(np.pi * X) * np.sin(np.pi * Y)

# Multiple t values along multiple y's
time_fracs = (0.0, 0.1, 0.3, 0.6, 1.0)  
y_slices = (0.0, 0.2, 0.5, 0.6, 1.0)

for y0 in y_slices:
    j0 = np.argmin(np.abs(y - y0))

    plt.figure(figsize=(15, 10))
    for frac in time_fracs:
        n = int(frac * (len(t) - 1))
        tval = t[n]

        U_num_slice = U[n, j0, :]          
        U_ex_full = u_exact(x, y, tval)
        U_ex_slice = U_ex_full[j0, :]

        plt.plot(x, U_num_slice,
                 label=f"FDM y={y[j0]:.2f}, t={tval:.2f}")
        plt.plot(x, U_ex_slice, 'k--', alpha=0.8,
                 label=f"Exact y={y[j0]:.2f}, t={tval:.2f}")

    plt.xlabel("x")
    plt.ylabel(f"u(x, y={y[j0]:.2f}, t)")
    plt.title(f"2D Heat Equation (FDM vs Analytic) at y ≈ {y0:.2f}")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Error at final time along multiple y values
U_exact_T = u_exact(x, y, T)

plt.figure(figsize=(15, 10))
for y0 in y_slices:
    j0 = np.argmin(np.abs(y - y0))
    err_slice = np.abs(U[-1, j0, :] - U_exact_T[j0, :])
    plt.plot(x, err_slice, label=f"y={y[j0]:.2f}")
plt.xlabel("x")
plt.ylabel("|Error|")
plt.title("Pointwise absolute error at final time (t=1) along multiple y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3D surface & heatmaps at final time
Xg, Yg = np.meshgrid(x, y, indexing='xy')
U_T = U[-1]

# 3D surface of u(x,y,T)
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, U_T, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y,T)')
ax.set_title('Numerical solution at t=1 (FDM)')
ax.view_init(elev=30, azim=225)
plt.tight_layout()
plt.show()

# Heatmap of u(x,y,T)
plt.figure(figsize=(12, 10))
plt.imshow(U_T, extent=[x[0], x[-1], y[0], y[-1]],
           origin='lower', aspect='equal', cmap='viridis')
plt.colorbar(label='u(x,y,T)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of u(x,y,T) (FDM)')
plt.tight_layout()
plt.show()

# Heatmap of |error| at t=1
Err_T = np.abs(U_T - U_exact_T)
plt.figure(figsize=(12, 10))
plt.imshow(Err_T, extent=[x[0], x[-1], y[0], y[-1]],
           origin='lower', aspect='equal', cmap='inferno')
plt.colorbar(label='|Error|')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of |U - U_exact| at t=1 (FDM)')
plt.tight_layout()
plt.show()
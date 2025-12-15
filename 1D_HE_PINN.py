# PINN solver for the 1D Heat Equation
#   u_t = D * u_xx
#   IC: u(x,0)=sin(pi x)
#   Dirichlet BCs: u(0,t)=u(1,t)=0

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import optax
import time

# Problem parameters
D = 0.1                      
xmin, xmax = 0.0, 1.0
tmin, tmax = 0.0, 1.0

# Training sizes
N_ic = 200                   # initial condition points
N_bc = 200                   # boundary condition points per side
N_r  = 10_000                # interior collocation points

key = jax.random.PRNGKey(0)

# Neural net utilities (MLP with tanh)
def init_params(layers, key):
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        limit = jnp.sqrt(6.0 / n_in)
        W = jax.random.uniform(k, (n_in, n_out), minval=-limit, maxval=limit)
        b = jnp.zeros((n_out,))
        params.append({'W': W, 'b': b})
    return params

def mlp_forward(params, t, x):
    H = jnp.concatenate([t, x], axis=1)  
    *hidden, last = params
    for lyr in hidden:
        H = jnp.tanh(H @ lyr['W'] + lyr['b'])
    return H @ last['W'] + last['b']      

# Autodiff: scalar -> vmap
def u_scalar(params, t, x):
    return mlp_forward(params, t=jnp.array([[t]]), x=jnp.array([[x]]))[0, 0]

du_dt   = jax.grad(lambda P, t, x: u_scalar(P, t, x), argnums=1)
du_dx   = jax.grad(lambda P, t, x: u_scalar(P, t, x), argnums=2)
d2u_dx2 = jax.grad(lambda P, t, x: du_dx(P, t, x), argnums=2)

# Vectorised versions
v_u        = jax.vmap(u_scalar,   in_axes=(None, 0, 0))
v_du_dt    = jax.vmap(du_dt,      in_axes=(None, 0, 0))
v_d2u_dx2  = jax.vmap(d2u_dx2,    in_axes=(None, 0, 0))

# Training data
def make_training_data(N_ic, N_bc, N_r, key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # IC: t=0, u(x,0) = sin(pi x)
    x_ic = jax.random.uniform(k1, (N_ic, 1), minval=xmin, maxval=xmax)
    t_ic = jnp.zeros_like(x_ic)
    u_ic = jnp.sin(jnp.pi * x_ic)

    # BCs: u(0,t)=0 and u(1,t)=0
    t_bc_left  = jax.random.uniform(k2, (N_bc, 1), minval=tmin, maxval=tmax)
    x_bc_left  = jnp.zeros_like(t_bc_left) + xmin
    u_bc_left  = jnp.zeros_like(t_bc_left)

    t_bc_right = jax.random.uniform(k3, (N_bc, 1), minval=tmin, maxval=tmax)
    x_bc_right = jnp.zeros_like(t_bc_right) + xmax
    u_bc_right = jnp.zeros_like(t_bc_right)

    # Interior collocation
    t_r = jax.random.uniform(k4, (N_r, 1), minval=tmin, maxval=tmax)
    x_r = jax.random.uniform(k5, (N_r, 1), minval=xmin, maxval=xmax)

    data = {
        'IC': (t_ic, x_ic, u_ic),                 # shapes (N_ic,1)
        'BC_L': (t_bc_left,  x_bc_left,  u_bc_left),
        'BC_R': (t_bc_right, x_bc_right, u_bc_right),
        'COL': (t_r, x_r)                         # shapes (N_r,1)
    }
    return data

data = make_training_data(N_ic, N_bc, N_r, key)

# Physics-informed loss
@jax.jit
def mse(a, b):
    return jnp.mean((a - b) ** 2)

# PDE Residual r = u_t - D u_xx 
def pde_residual(params, t, x):
    tt = t.flatten()
    xx = x.flatten()
    ut  = v_du_dt(params, tt, xx).reshape(-1, 1)
    uxx = v_d2u_dx2(params, tt, xx).reshape(-1, 1)
    return ut - D * uxx

def loss_fn(params, data, w_ic=1.0, w_bc=1.0, w_pde=1.0):
    # IC term
    t_ic, x_ic, u_ic = data['IC']
    uhat_ic = mlp_forward(params, t_ic, x_ic)
    L_ic = mse(uhat_ic, u_ic)

    # BC terms (Dirichlet both sides)
    tL, xL, uL = data['BC_L']
    tR, xR, uR = data['BC_R']
    uhatL = mlp_forward(params, tL, xL)
    uhatR = mlp_forward(params, tR, xR)
    L_bc = mse(uhatL, uL) + mse(uhatR, uR)

    # PDE residual in the interior
    t_r, x_r = data['COL']
    res = pde_residual(params, t_r, x_r)
    L_pde = mse(res, jnp.zeros_like(res))

    return w_ic * L_ic + w_bc * L_bc + w_pde * L_pde

# Optimizer + update
lr_sched = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={8000: 0.5, 16000: 0.2, 24000: 0.2}
)
optimizer = optax.adam(lr_sched)

# Initialise network
params = init_params([2] + [32]*4 + [1], key)
opt_state = optimizer.init(params)

@jax.jit
def update(opt_state, params, data):
    grads = jax.grad(loss_fn)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

# Train
epochs = 20000  
print_every = 100
loss_hist = []

plt.rcParams.update({
    'font.size': 16,         
    'axes.titlesize': 20,    
    'axes.labelsize': 18,    
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

print("Training PINN for 1D heat equation...")
t0 = time.time()
for ep in range(epochs):
    opt_state, params = update(opt_state, params, data)
    if ep % print_every == 0:
        L = float(loss_fn(params, data))
        loss_hist.append(L)
        print(f"Epoch {ep:5d} | Loss: {L:.3e}")
t1 = time.time()
print(f"Training time: {t1 - t0:.2f} s")

# Analytic solution
def u_exact_fun(x, t):
    return jnp.exp(-jnp.pi**2 * D * t) * jnp.sin(jnp.pi * x)

# Time-slice comparisons vs analytic
N_plot = 400
x_plot = jnp.linspace(xmin, xmax, N_plot).reshape(-1, 1)
times_to_show = [0.0, 0.1, 0.3, 0.6, 1.0]

plt.figure(figsize=(15,10))
for tt in times_to_show:
    t_plot = jnp.ones_like(x_plot) * tt
    u_pred = mlp_forward(params, t_plot, x_plot)
    u_true = u_exact_fun(x_plot, tt)
    plt.plot(np.array(x_plot), np.array(u_pred), label=f"PINN t={tt:.2f}")
    plt.plot(np.array(x_plot), np.array(u_true), 'k--', alpha=0.8, label=f"Exact t={tt:.2f}")
plt.xlabel('x'); plt.ylabel('u(x,t)')
plt.title('1D Heat Equation (PINN vs Analytic)')
plt.legend(ncol=2); plt.grid(True); plt.tight_layout(); plt.show()

# Final-time comparison and error profile
t_final = tmax
t_plot_fin = jnp.ones_like(x_plot) * t_final
u_pred_fin = mlp_forward(params, t_plot_fin, x_plot)
u_true_fin = u_exact_fun(x_plot, t_final)
err_fin = jnp.abs(u_pred_fin - u_true_fin)

plt.figure(figsize=(8,5))
plt.plot(np.array(x_plot), np.array(u_true_fin), 'k--', label='Analytic')
plt.plot(np.array(x_plot), np.array(u_pred_fin), 'r',   label='PINN')
plt.xlabel('x'); plt.ylabel('u(x,T)')
plt.title(f'Final time t={t_final}: PINN vs Analytic')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(15,10))
plt.plot(np.array(x_plot), np.array(err_fin))
plt.xlabel('x'); plt.ylabel('|Error|')
plt.title('Pointwise absolute error at t=1')
plt.grid(True); plt.tight_layout(); plt.show()

# 3D surface of u(x,t) and error surface
nx, nt = 60, 60
xg = jnp.linspace(xmin, xmax, nx)
tg = jnp.linspace(tmin, tmax, nt)
Tg, Xg = jnp.meshgrid(tg, xg, indexing='xy')  

TT = Tg.reshape(-1, 1)
XX = Xg.reshape(-1, 1)
U_flat = mlp_forward(params, TT, XX)          
U = np.array(U_flat).reshape(nx, nt)

# Analytic on the same grid
U_exact = np.array(u_exact_fun(XX, TT)).reshape(nx, nt)
Err = np.abs(U - U_exact)

# 3D surface u(x,t)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Tg), np.array(Xg), U, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('x'); ax.set_zlabel('u')
ax.set_title('Numerical solution u(x,t) (PINN)')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# 3D surface |error|
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Tg), np.array(Xg), Err, cmap=cm.inferno, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('x'); ax.set_zlabel('|Error|')
ax.set_title('Pointwise absolute error |U_PINN - U_exact|')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# Heatmaps
plt.figure(figsize=(15,12))
plt.imshow(U, extent=[tmin, tmax, xmin, xmax], aspect='auto',
           origin='lower', cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('t'); plt.ylabel('x')
plt.title('Heatmap of u(x,t) (PINN)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(15,12))
plt.imshow(Err, extent=[tmin, tmax, xmin, xmax], aspect='auto',
           origin='lower', cmap='inferno')
plt.colorbar(label='|Error|')
plt.xlabel('t'); plt.ylabel('x')
plt.title('Heatmap of |U_PINN - U_exact|')
plt.tight_layout(); plt.show()

# Training loss
plt.figure(figsize=(8,5))
plt.semilogy(np.arange(0, len(loss_hist))*print_every, loss_hist)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training loss'); plt.grid(True); plt.tight_layout(); plt.show()
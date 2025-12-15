#2D Heat Equation PINN with hard IC and BC
#PDE: u_t = D (u_xx + u_yy),     (x,y) in (0,1)x(0,1), t in (0,1]
#IC (hard-encoded): u(x,y,0) = sin(pi x) sin(pi y)
#BC (hard-encoded, Dirichlet): u(0,y,t) = u(1,y,t) = u(x,0,t) = u(x,1,t) = 0
#Analytic solution: u_exact(x,y,t) = exp(-2*pi^2*D*t) * sin(pi x) sin(pi y)

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import optax

# Problem parameters
D = 0.1
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
tmin, tmax = 0.0, 1.0

# Collocation points for PDE residual
N_r = 8000  

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

def mlp_forward(params, t, x, y):
    H = jnp.concatenate([t, x, y], axis=1)  
    *hidden, last = params
    for lyr in hidden:
        H = jnp.tanh(H @ lyr['W'] + lyr['b'])
    return H @ last['W'] + last['b']        

# Ansatz with hard IC and BC
def u_ansatz_scalar(params, t, x, y):
    N_val = mlp_forward(
        params,
        t=jnp.array([[t]]),
        x=jnp.array([[x]]),
        y=jnp.array([[y]])
    )[0, 0]
    base = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    envelope = t * x * (1.0 - x) * y * (1.0 - y)
    return base + envelope * N_val

# Derivatives of ansatz u
du_dt   = jax.grad(lambda P, t, x, y: u_ansatz_scalar(P, t, x, y), argnums=1)
du_dx   = jax.grad(lambda P, t, x, y: u_ansatz_scalar(P, t, x, y), argnums=2)
du_dy   = jax.grad(lambda P, t, x, y: u_ansatz_scalar(P, t, x, y), argnums=3)
d2u_dx2 = jax.grad(lambda P, t, x, y: du_dx(P, t, x, y), argnums=2)
d2u_dy2 = jax.grad(lambda P, t, x, y: du_dy(P, t, x, y), argnums=3)

# vectorised versions
v_u        = jax.vmap(u_ansatz_scalar, in_axes=(None, 0, 0, 0))
v_du_dt    = jax.vmap(du_dt,           in_axes=(None, 0, 0, 0))
v_d2u_dx2  = jax.vmap(d2u_dx2,         in_axes=(None, 0, 0, 0))
v_d2u_dy2  = jax.vmap(d2u_dy2,         in_axes=(None, 0, 0, 0))

# Collocation points (interior only)
def make_collocation_points(N_r, key):
    k1, k2, k3 = jax.random.split(key, 3)

    t_r = jax.random.uniform(k1, (N_r, 1), minval=tmin, maxval=tmax)  
    x_r = jax.random.uniform(k2, (N_r, 1), minval=xmin, maxval=xmax) 
    y_r = jax.random.uniform(k3, (N_r, 1), minval=ymin, maxval=ymax) 

    return t_r, x_r, y_r

t_r, x_r, y_r = make_collocation_points(N_r, key)

# PDE residual & loss
@jax.jit
def mse(a, b):
    return jnp.mean((a - b) ** 2)

#PDE Residual r = u_t - D (u_xx + u_yy)
def pde_residual(params, t, x, y):
    tt = t.flatten()
    xx = x.flatten()
    yy = y.flatten()
    ut  = v_du_dt(params, tt, xx, yy).reshape(-1, 1)
    uxx = v_d2u_dx2(params, tt, xx, yy).reshape(-1, 1)
    uyy = v_d2u_dy2(params, tt, xx, yy).reshape(-1, 1)
    return ut - D * (uxx + uyy)

@jax.jit
def loss_fn(params, t_r, x_r, y_r):
    res = pde_residual(params, t_r, x_r, y_r)
    return mse(res, jnp.zeros_like(res))

# Optimizer + update
lr_sched = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={8000: 0.5, 16000: 0.2, 24000: 0.2}
)
optimizer = optax.adam(lr_sched)

layers = [3] + [32]*4 + [1]    
params = init_params(layers, key)
opt_state = optimizer.init(params)

@jax.jit
def update(opt_state, params, t_r, x_r, y_r):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, t_r, x_r, y_r)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss_val

# Train
epochs = 20000
print_every = 200
loss_hist = []

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

print("Training hard-IC/BC PINN for 2D heat equation...")
t0 = time.time()
for ep in range(epochs):
    opt_state, params, L = update(opt_state, params, t_r, x_r, y_r)
    if ep % print_every == 0:
        loss_hist.append(float(L))
        print(f"Epoch {ep:5d} | PDE Loss: {float(L):.3e}")
t1 = time.time()
print(f"Training time: {t1 - t0:.2f} s")

# Analytic solution
def u_exact_fun(x, y, t):
    return jnp.exp(-2.0 * jnp.pi**2 * D * t) * \
           jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

# Loss curve
epochs_logged = np.arange(0, len(loss_hist)) * print_every
plt.figure(figsize=(8,5))
plt.semilogy(epochs_logged, loss_hist, label='PDE loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training loss (PDE residual only)')
plt.grid(True, which='both'); plt.tight_layout(); plt.show()

# Time-slice comparisons vs analytic at y=0.5
N_plot = 400
x_plot = jnp.linspace(xmin, xmax, N_plot).reshape(-1, 1)
y0 = 0.5
y_plot = jnp.ones_like(x_plot) * y0
times_to_show = [0.0, 0.1, 0.3, 0.6, 1.0]

plt.figure(figsize=(15,10))
for tt in times_to_show:
    t_plot = jnp.ones_like(x_plot) * tt
    u_pred = v_u(params, t_plot.flatten(), x_plot.flatten(), y_plot.flatten()).reshape(-1,1)
    u_true = u_exact_fun(x_plot, y_plot, tt)
    plt.plot(np.array(x_plot), np.array(u_pred), label=f"PINN t={tt:.2f}")
    plt.plot(np.array(x_plot), np.array(u_true), 'k--', alpha=0.8,
             label=f"Exact t={tt:.2f}")
plt.xlabel('x'); plt.ylabel(f'u(x, y={y0:.2f}, t)')
plt.title('2D Heat Equation (PINN vs Analytic) at y=0.5 (hard IC/BC)')
plt.legend(ncol=2); plt.grid(True); plt.tight_layout(); plt.show()

# Final-time comparison & error at y=0.5
t_final = tmax
t_plot_fin = jnp.ones_like(x_plot) * t_final
u_pred_fin = v_u(params, t_plot_fin.flatten(),
                 x_plot.flatten(), y_plot.flatten()).reshape(-1,1)
u_true_fin = u_exact_fun(x_plot, y_plot, t_final)
err_fin = jnp.abs(u_pred_fin - u_true_fin)

plt.figure(figsize=(8,5))
plt.plot(np.array(x_plot), np.array(u_true_fin), 'k--', label='Analytic')
plt.plot(np.array(x_plot), np.array(u_pred_fin), 'r',   label='PINN')
plt.xlabel('x'); plt.ylabel('u(x, y=0.5, T)')
plt.title(f'Final time t={t_final}: PINN vs Analytic (y=0.5)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(15,10))
plt.plot(np.array(x_plot), np.array(err_fin))
plt.xlabel('x'); plt.ylabel('|Error|')
plt.title('Pointwise absolute error at t=1 (y=0.5)')
plt.grid(True); plt.tight_layout(); plt.show()

# 3D surface & heatmaps at t=1
nx, ny = 60, 60
xg = jnp.linspace(xmin, xmax, nx)
yg = jnp.linspace(ymin, ymax, ny)
Xg, Yg = jnp.meshgrid(xg, yg, indexing='xy')

Tg = jnp.ones_like(Xg) * t_final
TT = Tg.reshape(-1)
XX = Xg.reshape(-1)
YY = Yg.reshape(-1)

U_flat = v_u(params, TT, XX, YY)
U = np.array(U_flat).reshape(ny, nx)

U_exact = np.array(u_exact_fun(XX, YY, t_final)).reshape(ny, nx)
Err = np.abs(U - U_exact)

# 3D surface u(x,y,t = 1.0)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Xg), np.array(Yg), U,
                cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u(x,y,T)')
ax.set_title('Numerical solution u(x,y,T) (PINN, hard IC/BC)')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# 3D surface |error|
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Xg), np.array(Yg), Err,
                cmap=cm.inferno, linewidth=0, antialiased=True)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('|Error|')
ax.set_title('Pointwise absolute error |U_PINN - U_exact|')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# Heatmaps
plt.figure(figsize=(15,12))
plt.imshow(U, extent=[xmin, xmax, ymin, ymax], aspect='equal',
           origin='lower', cmap='viridis')
plt.colorbar(label='u(x,y,T)'); plt.xlabel('x'); plt.ylabel('y')
plt.title('Heatmap of u(x,y,T) (PINN, hard IC/BC)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(15,12))
plt.imshow(Err, extent=[xmin, xmax, ymin, ymax], aspect='equal',
           origin='lower', cmap='inferno')
plt.colorbar(label='|U_PINN - U_exact|'); plt.xlabel('x'); plt.ylabel('y')
plt.title('Heatmap of |U_PINN - U_exact| at t=1 (hard IC/BC)')
plt.tight_layout(); plt.show()
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import optax
import time
from jax.scipy.special import erf  

# Black–Scholes parameters
r = 0.05          # risk-free interest rate
sigma = 0.2       # volatility
K = 1.0           # strike
T = 1.0           # maturity

Smin, Smax = 0.0, 4.0 * K   # underlying price domain
tmin, tmax = 0.0, T         # time domain [0, T]

# Training sizes
N_ic = 200       # terminal-condition points (at t=T)
N_bc = 200       # boundary condition points per side
N_r  = 10_000    # interior collocation points

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

def mlp_forward(params, t, S):
    H = jnp.concatenate([t, S], axis=1)  
    *hidden, last = params
    for lyr in hidden:
        H = jnp.tanh(H @ lyr['W'] + lyr['b'])
    return H @ last['W'] + last['b']     

# Autodiff: scalar -> vmap
def V_scalar(params, t, S):
    return mlp_forward(params,
                       t=jnp.array([[t]]),
                       S=jnp.array([[S]]))[0, 0]

dV_dt   = jax.grad(lambda P, t, S: V_scalar(P, t, S), argnums=1)
dV_dS   = jax.grad(lambda P, t, S: V_scalar(P, t, S), argnums=2)
d2V_dS2 = jax.grad(lambda P, t, S: dV_dS(P, t, S), argnums=2)

# Vectorised versions
v_V       = jax.vmap(V_scalar,   in_axes=(None, 0, 0))
v_dV_dt   = jax.vmap(dV_dt,      in_axes=(None, 0, 0))
v_dV_dS   = jax.vmap(dV_dS,      in_axes=(None, 0, 0))
v_d2V_dS2 = jax.vmap(d2V_dS2,    in_axes=(None, 0, 0))

# Training data
def make_training_data(N_ic, N_bc, N_r, key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Terminal condition at t = T: V(S,T) = max(S - K, 0)
    S_ic = jax.random.uniform(k1, (N_ic, 1), minval=Smin, maxval=Smax)
    t_ic = jnp.ones_like(S_ic) * T
    V_ic = jnp.maximum(S_ic - K, 0.0)

    # BC at S = 0: V(0,t) = 0
    t_bc_left  = jax.random.uniform(k2, (N_bc, 1), minval=tmin, maxval=tmax)
    S_bc_left  = jnp.zeros_like(t_bc_left) + Smin
    V_bc_left  = jnp.zeros_like(t_bc_left)

    # BC at S = Smax: V(Smax,t) ~ Smax - K*exp(-r*(T - t))
    t_bc_right = jax.random.uniform(k3, (N_bc, 1), minval=tmin, maxval=tmax)
    S_bc_right = jnp.zeros_like(t_bc_right) + Smax
    V_bc_right = Smax - K * jnp.exp(-r * (T - t_bc_right))

    # Interior collocation points
    t_r = jax.random.uniform(k4, (N_r, 1), minval=tmin, maxval=tmax)
    S_r = jax.random.uniform(k5, (N_r, 1), minval=Smin, maxval=Smax)

    data = {
        'IC':   (t_ic, S_ic, V_ic),                    # shapes (N_ic,1)
        'BC_L': (t_bc_left,  S_bc_left,  V_bc_left),
        'BC_R': (t_bc_right, S_bc_right, V_bc_right),
        'COL':  (t_r, S_r)                             # shapes (N_r,1)
    }
    return data

data = make_training_data(N_ic, N_bc, N_r, key)

# Physics-informed loss
@jax.jit
def mse(a, b):
    return jnp.mean((a - b) ** 2)

# PDE Residual r = V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V
def pde_residual(params, t, S):
    tt = t.flatten()
    Ss = S.flatten()
    Vt  = v_dV_dt(params,  tt, Ss).reshape(-1, 1)
    VS  = v_dV_dS(params,  tt, Ss).reshape(-1, 1)
    VSS = v_d2V_dS2(params, tt, Ss).reshape(-1, 1)
    Vval = v_V(params, tt, Ss).reshape(-1, 1)

    return Vt + 0.5 * sigma**2 * (Ss.reshape(-1,1)**2) * VSS \
             + r * Ss.reshape(-1,1) * VS - r * Vval

def loss_fn(params, data, w_ic=1.0, w_bc=1.0, w_pde=1.0):
    # Terminal condition term (IC in time-reversed)
    t_ic, S_ic, V_ic = data['IC']
    Vhat_ic = mlp_forward(params, t_ic, S_ic)
    L_ic = mse(Vhat_ic, V_ic)

    # BC terms
    tL, SL, VL = data['BC_L']
    tR, SR, VR = data['BC_R']
    VhatL = mlp_forward(params, tL, SL)
    VhatR = mlp_forward(params, tR, SR)
    L_bc = mse(VhatL, VL) + mse(VhatR, VR)

    # PDE residual in the interior
    t_r, S_r = data['COL']
    res = pde_residual(params, t_r, S_r)
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

# Analytic Black–Scholes solution for comparison
def N_cdf(x):
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

def bs_call_analytic(S, t):
    S = jnp.asarray(S)
    t = jnp.asarray(t)
    tau = T - t  # time to maturity

    S_safe = jnp.maximum(S, 1e-16)
    tau_pos = jnp.maximum(tau, 1e-8)

    d1 = (jnp.log(S_safe / K) + (r + 0.5 * sigma**2) * tau_pos) / (sigma * jnp.sqrt(tau_pos))
    d2 = d1 - sigma * jnp.sqrt(tau_pos)

    value_bs = S_safe * N_cdf(d1) - K * jnp.exp(-r * tau_pos) * N_cdf(d2)
    payoff = jnp.maximum(S - K, 0.0)

    return jnp.where(tau <= 0.0, payoff, value_bs)

def V_exact_fun(S, t):
    return bs_call_analytic(S, t)

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

print("Training PINN for 1D Black–Scholes PDE (European call)...")
t0_train = time.time()
for ep in range(epochs):
    opt_state, params = update(opt_state, params, data)
    if ep % print_every == 0:
        L = float(loss_fn(params, data))
        loss_hist.append(L)
        print(f"Epoch {ep:5d} | Loss: {L:.3e}")
t1_train = time.time()
print(f"Training time: {t1_train - t0_train:.2f} s")

# Accuracy check at a single point
S0 = 1.3
t0 = 0.4

V_pinn_point = float(mlp_forward(params,
                                 jnp.array([[t0]]),
                                 jnp.array([[S0]]))[0, 0])
V_exact_point = float(V_exact_fun(S0, t0))
err_point = abs(V_pinn_point - V_exact_point)

print("\n--- Pointwise accuracy check (PINN) ---")
print(f"S0 = {S0}, t0 = {t0}")
print(f"PINN  : {V_pinn_point:.8f}")
print(f"Exact : {V_exact_point:.8f}")
print(f"Error : {err_point:.3e}")

# Time-slice comparisons vs analytic
N_plot = 400
S_plot = jnp.linspace(Smin, Smax, N_plot).reshape(-1, 1)
times_to_show = [0.0, 0.1, 0.3, 0.6, 1.0]

plt.figure(figsize=(15,10))
for tt in times_to_show:
    t_plot = jnp.ones_like(S_plot) * tt
    V_pred = mlp_forward(params, t_plot, S_plot)
    V_true = V_exact_fun(S_plot, tt)
    plt.plot(np.array(S_plot), np.array(V_pred), label=f"PINN t={tt:.2f}")
    plt.plot(np.array(S_plot), np.array(V_true), 'k--', alpha=0.8, label=f"Exact t={tt:.2f}")
plt.xlabel('S'); plt.ylabel('V(S,t)')
plt.title('Black–Scholes European Call (PINN vs Analytic)')
plt.legend(ncol=2); plt.grid(True); plt.tight_layout(); plt.show()

# Final-time comparison and error profile at t=T (today)
t_final = tmin
t_plot_fin = jnp.ones_like(S_plot) * t_final
V_pred_fin = mlp_forward(params, t_plot_fin, S_plot)
V_true_fin = V_exact_fun(S_plot, t_final)
err_fin = jnp.abs(V_pred_fin - V_true_fin)

plt.figure(figsize=(8,5))
plt.plot(np.array(S_plot), np.array(V_true_fin), 'k--', label='Analytic')
plt.plot(np.array(S_plot), np.array(V_pred_fin), 'r',   label='PINN')
plt.xlabel('S'); plt.ylabel('V(S,0)')
plt.title(f'Option value at t={t_final}: PINN vs Analytic')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(15,10))
plt.plot(np.array(S_plot), np.array(err_fin))
plt.xlabel('S'); plt.ylabel('|Error|')
plt.title('Pointwise absolute error at t=0')
plt.grid(True); plt.tight_layout(); plt.show()

# 3D surface of V(S,t) and error surface
nS, nt = 60, 60
Sg = jnp.linspace(Smin, Smax, nS)
tg = jnp.linspace(tmin, tmax, nt)
Tg, Sg_mesh = jnp.meshgrid(tg, Sg, indexing='xy')  # shapes (nS, nt)

TT = Tg.reshape(-1, 1)
SS = Sg_mesh.reshape(-1, 1)
V_flat = mlp_forward(params, TT, SS)         
V_num = np.array(V_flat).reshape(nS, nt)

# Analytic on the same grid
V_exact_grid = np.array(V_exact_fun(SS, TT)).reshape(nS, nt)
Err = np.abs(V_num - V_exact_grid)

# Global error metrics 
L2_err = float(np.sqrt(np.mean(Err**2)))
Linf_err = float(np.max(Err))

print("\n--- Global error metrics on 2D grid (PINN) ---")
print(f"L2   error = {L2_err:.3e}")
print(f"L_inf error = {Linf_err:.3e}")

# 3D surface V(S,t)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Tg), np.array(Sg_mesh), V_num, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('S'); ax.set_zlabel('V')
ax.set_title('Numerical solution V(S,t) (PINN, Black–Scholes)')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# 3D surface |error|
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(Tg), np.array(Sg_mesh), Err, cmap=cm.inferno, linewidth=0, antialiased=True)
ax.set_xlabel('t'); ax.set_ylabel('S'); ax.set_zlabel('|Error|')
ax.set_title('Pointwise absolute error |V_PINN - V_exact|')
ax.view_init(elev=30, azim=225)
plt.tight_layout(); plt.show()

# Heatmap of V(S,t) 
plt.figure(figsize=(15,12))
plt.imshow(V_num, extent=[tmin, tmax, Smin, Smax], aspect='auto',
           origin='lower', cmap='viridis')
plt.colorbar(label='V(S,t)')
plt.xlabel('t'); plt.ylabel('S')
plt.title('Heatmap of V(S,t) (PINN, Black–Scholes)')
plt.tight_layout(); plt.show()

# Heatmap of |error|
plt.figure(figsize=(15,12))
plt.imshow(Err, extent=[tmin, tmax, Smin, Smax], aspect='auto',
           origin='lower', cmap='inferno')
plt.colorbar(label='|Error|')
plt.xlabel('t'); plt.ylabel('S')
plt.title('Heatmap of |V_PINN - V_exact|')
plt.tight_layout(); plt.show()

# Training loss
plt.figure(figsize=(8,5))
plt.semilogy(np.arange(0, len(loss_hist))*print_every, loss_hist)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training loss'); plt.grid(True); plt.tight_layout(); plt.show()
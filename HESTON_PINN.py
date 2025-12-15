import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

#  Heston parameters
K     = 100.0  # strike
r     = 0.02   # risk-free rate
q     = 0.05   # dividend yield
T     = 0.15   # maturity
sigma = 0.3    # vol-of-variance
rho   = -0.9   # correlation
kappa = 1.5    # mean reversion speed
theta = 0.04   # long-run variance

Smin, Smax = 0.0, 2.0 * K   # S in [0, 200]
vmin, vmax = 0.0, 0.5       # v in [0, 0.5]

# evaluation point 
S0 = 101.52
v0 = 0.05412

key = jax.random.PRNGKey(0)

def init_params(layers, key):
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        limit = jnp.sqrt(6.0 / n_in)
        W = jax.random.uniform(k, (n_in, n_out), minval=-limit, maxval=limit)
        b = jnp.zeros((n_out,))
        params.append({'W': W, 'b': b})
    return params

def mlp_forward(params, t, S, v):
    H = jnp.concatenate([t, S, v], axis=1) 
    *hidden, last = params
    for lyr in hidden:
        H = jnp.tanh(H @ lyr['W'] + lyr['b'])
    return H @ last['W'] + last['b']

# scalar version for autodiff
def U_scalar(params, t, S, v):
    return mlp_forward(
        params,
        t=jnp.array([[t]]),
        S=jnp.array([[S]]),
        v=jnp.array([[v]])
    )[0, 0]



# AUTODIFF DERIVATIVES
# First-order derivatives
dU_dt = jax.grad(lambda P, t, S, v: U_scalar(P, t, S, v), argnums=1)
dU_dS = jax.grad(lambda P, t, S, v: U_scalar(P, t, S, v), argnums=2)
dU_dv = jax.grad(lambda P, t, S, v: U_scalar(P, t, S, v), argnums=3)

# Second-order derivatives
d2U_dS2 = jax.grad(lambda P, t, S, v: dU_dS(P, t, S, v), argnums=2)
d2U_dv2 = jax.grad(lambda P, t, S, v: dU_dv(P, t, S, v), argnums=3)

# Mixed derivative U_Sv
d2U_dSdv = jax.grad(lambda P, t, S, v: dU_dS(P, t, S, v), argnums=3)

# Vectorised versions 
v_U        = jax.vmap(U_scalar,      in_axes=(None, 0, 0, 0))
v_dU_dt    = jax.vmap(dU_dt,         in_axes=(None, 0, 0, 0))
v_dU_dS    = jax.vmap(dU_dS,         in_axes=(None, 0, 0, 0))
v_dU_dv    = jax.vmap(dU_dv,         in_axes=(None, 0, 0, 0))
v_d2U_dS2  = jax.vmap(d2U_dS2,       in_axes=(None, 0, 0, 0))
v_d2U_dv2  = jax.vmap(d2U_dv2,       in_axes=(None, 0, 0, 0))
v_d2U_dSdv = jax.vmap(d2U_dSdv,      in_axes=(None, 0, 0, 0))

# Training data (IC, BC, PDE)
def make_training_data(key,
                       N_ic=3000,
                       N_bc_S0=2000,
                       N_bc_Smax=2000,
                       N_bc_v0=2000,
                       N_bc_vmax=2000,
                       N_r=20000):
    
#def make_training_data(key,
 #                      N_ic=2000,
  #                     N_bc_S0=1000,
   #                    N_bc_Smax=1000,
    #                   N_bc_v0=1000,
     #                  N_bc_vmax=1000,
      #                 N_r=8000):

    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    # Terminal condition at t = T (U(S,v,T) = max(S-K, 0))
    S_ic = jax.random.uniform(k1, (N_ic, 1), minval=Smin, maxval=Smax)
    v_ic = jax.random.uniform(k2, (N_ic, 1), minval=vmin, maxval=vmax)
    t_ic = jnp.ones_like(S_ic) * T
    U_ic = jnp.maximum(S_ic - K, 0.0)

    # BC at S = 0: U(0,v,t) = 0 
    t_bc_S0 = jax.random.uniform(k3, (N_bc_S0, 1), minval=0.0, maxval=T)
    v_bc_S0 = jax.random.uniform(k4, (N_bc_S0, 1), minval=vmin, maxval=vmax)
    S_bc_S0 = jnp.zeros_like(t_bc_S0) + Smin
    U_bc_S0 = jnp.zeros_like(t_bc_S0)

    # BC at S = Smax: large-S asymptotic with dividends
    t_bc_Smax = jax.random.uniform(k5, (N_bc_Smax, 1), minval=0.0, maxval=T)
    v_bc_Smax = jax.random.uniform(k6, (N_bc_Smax, 1), minval=vmin, maxval=vmax)
    S_bc_Smax = jnp.zeros_like(t_bc_Smax) + Smax
    tau_bc_Smax = T - t_bc_Smax
    U_bc_Smax = Smax * jnp.exp(-q * tau_bc_Smax) - K * jnp.exp(-r * tau_bc_Smax)

    # Neumann BC at v = 0: U_v(S, 0, t) = 0 
    t_bc_v0 = jax.random.uniform(k7, (N_bc_v0, 1), minval=0.0, maxval=T)
    S_bc_v0 = jax.random.uniform(k1, (N_bc_v0, 1), minval=Smin, maxval=Smax)
    v_bc_v0 = jnp.zeros_like(t_bc_v0) + vmin

    # Neumann BC at v = vmax: U_v(S, vmax, t) = 0 
    t_bc_vmax = jax.random.uniform(k2, (N_bc_vmax, 1), minval=0.0, maxval=T)
    S_bc_vmax = jax.random.uniform(k3, (N_bc_vmax, 1), minval=Smin, maxval=Smax)
    v_bc_vmax = jnp.zeros_like(t_bc_vmax) + vmax

    # Interior collocation points for PDE 
    t_r = jax.random.uniform(k4, (N_r, 1), minval=0.0, maxval=T)
    S_r = jax.random.uniform(k5, (N_r, 1), minval=Smin, maxval=Smax)
    v_r = jax.random.uniform(k6, (N_r, 1), minval=vmin, maxval=vmax)

    data = {
        'IC': (t_ic, S_ic, v_ic, U_ic),
        'BC_S0': (t_bc_S0, S_bc_S0, v_bc_S0, U_bc_S0),
        'BC_Smax': (t_bc_Smax, S_bc_Smax, v_bc_Smax, U_bc_Smax),
        'BC_v0': (t_bc_v0, S_bc_v0, v_bc_v0),
        'BC_vmax': (t_bc_vmax, S_bc_vmax, v_bc_vmax),
        'COL': (t_r, S_r, v_r)
    }
    return data


data = make_training_data(key)


# Physics informed loss
@jax.jit
def mse(a, b):
    return jnp.mean((a - b)**2)

# PDE Residual r = U_t + 0.5*v*S^2*U_SS + rho*sigma*v*S*U_Sv + 0.5*sigma^2*v*U_vv + (r - q)*S*U_S + kappa*(theta - v)*U_v - r*U
def pde_residual(params, t, S, v):
    tt = t.flatten()
    Ss = S.flatten()
    vv = v.flatten()

    Ut   = v_dU_dt(params, tt, Ss, vv)
    US   = v_dU_dS(params, tt, Ss, vv)
    Uv   = v_dU_dv(params, tt, Ss, vv)
    USS  = v_d2U_dS2(params, tt, Ss, vv)
    Uvv  = v_d2U_dv2(params, tt, Ss, vv)
    USv  = v_d2U_dSdv(params, tt, Ss, vv)
    Uval = v_U(params, tt, Ss, vv)

    Ut   = Ut.reshape(-1, 1)
    US   = US.reshape(-1, 1)
    Uv   = Uv.reshape(-1, 1)
    USS  = USS.reshape(-1, 1)
    Uvv  = Uvv.reshape(-1, 1)
    USv  = USv.reshape(-1, 1)
    Uval = Uval.reshape(-1, 1)
    Ss   = Ss.reshape(-1, 1)
    vv   = vv.reshape(-1, 1)

    res = (Ut + 0.5 * vv * Ss**2 * USS
        + rho * sigma * vv * Ss * USv
        + 0.5 * sigma**2 * vv * Uvv
        + (r - q) * Ss * US
        + kappa * (theta - vv) * Uv
        - r * Uval
    )

    return res


def loss_fn(params, data,
            #w_ic=1.0, w_bc_S=1.0, w_bc_v=1.0, w_pde=1.0):
            w_ic=10.0, w_bc_S=5.0, w_bc_v=5.0, w_pde=1.0):
    # Terminal condition
    t_ic, S_ic, v_ic, U_ic = data['IC']
    Uhat_ic = mlp_forward(params, t_ic, S_ic, v_ic)
    L_ic = mse(Uhat_ic, U_ic)

    # S = 0 boundary
    tS0, SS0, vS0, US0 = data['BC_S0']
    Uhat_S0 = mlp_forward(params, tS0, SS0, vS0)
    L_bc_S0 = mse(Uhat_S0, US0)

    # S = Smax boundary
    tSm, SSm, vSm, USm = data['BC_Smax']
    Uhat_Smax = mlp_forward(params, tSm, SSm, vSm)
    L_bc_Smax = mse(Uhat_Smax, USm)

    L_bc_S = L_bc_S0 + L_bc_Smax

    # v = 0 Neumann: U_v = 0
    tv0, Sv0, vv0 = data['BC_v0']
    Uv_v0 = v_dU_dv(params, tv0.flatten(), Sv0.flatten(), vv0.flatten())
    L_bc_v0 = mse(Uv_v0.reshape(-1, 1), jnp.zeros_like(Uv_v0).reshape(-1, 1))

    # v = vmax Neumann: U_v = 0
    tvM, SvM, vvM = data['BC_vmax']
    Uv_vM = v_dU_dv(params, tvM.flatten(), SvM.flatten(), vvM.flatten())
    L_bc_vM = mse(Uv_vM.reshape(-1, 1), jnp.zeros_like(Uv_vM).reshape(-1, 1))

    L_bc_v = L_bc_v0 + L_bc_vM

    # PDE residual in interior
    t_r, S_r, v_r = data['COL']
    res = pde_residual(params, t_r, S_r, v_r)
    L_pde = mse(res, jnp.zeros_like(res))

    return (w_ic * L_ic
            + w_bc_S * L_bc_S
            + w_bc_v * L_bc_v
            + w_pde * L_pde)


# Optimizer and training loop
#layers = [3] + [12]*4 + [1]  
layers = [3] + [64]*4 + [1]   
params = init_params(layers, key)

lr_sched = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={5000: 0.5, 10000: 0.2, 15000: 0.2}
)
optimizer = optax.adam(lr_sched)
opt_state = optimizer.init(params)

@jax.jit
def update(opt_state, params, data):
    grads = jax.grad(loss_fn)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

epochs = 20000 #5000
print_every = 500
loss_hist = []

print("Training PINN for Heston PDE with Neumann v-BCs...")
t0_train = time.time()
for ep in range(epochs):
    opt_state, params = update(opt_state, params, data)
    if ep % print_every == 0:
        L = float(loss_fn(params, data))
        loss_hist.append(L)
        print(f"Epoch {ep:5d} | Loss: {L:.3e}")
t1_train = time.time()
print(f"Training time: {t1_train - t0_train:.2f} s")

# Pointwise price at t=0
U_pinn_point = float(
    mlp_forward(params,
                t=jnp.array([[0.0]]),
                S=jnp.array([[S0]]),
                v=jnp.array([[v0]]))[0, 0]
)
print("\n--- PINN price at (t=0, S0, v0) ---")
print(f"S0 = {S0}, v0 = {v0}")
print(f"PINN price: {U_pinn_point:.8f}")

# Create a grid for t=0 surface: U(S,v, t=0)
nS, nv = 60, 60
S_grid = jnp.linspace(Smin, Smax, nS).reshape(-1, 1)
v_grid = jnp.linspace(vmin, vmax, nv).reshape(-1, 1)

Sg, vg = jnp.meshgrid(S_grid.flatten(), v_grid.flatten(), indexing='ij')
TT0 = jnp.zeros_like(Sg)    

TT_flat = TT0.reshape(-1, 1)
SS_flat = Sg.reshape(-1, 1)
vv_flat = vg.reshape(-1, 1)

U_flat = mlp_forward(params, TT_flat, SS_flat, vv_flat)
U_surface = np.array(U_flat).reshape(nS, nv)

# Plot surface
Sg_np = np.array(Sg)
vg_np = np.array(vg)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Sg_np, vg_np, U_surface, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('S')
ax.set_ylabel('v')
ax.set_zlabel('U(S,v,t=0)')
ax.set_title('Heston PINN solution at t=0 (Neumann v-BCs)')
ax.view_init(elev=30, azim=225)
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(U_surface.T,
           extent=[Smin, Smax, vmin, vmax],
           origin='lower',
           aspect='auto',
           cmap='viridis')
plt.colorbar(label='U(S,v,t=0)')
plt.xlabel('S')
plt.ylabel('v')
plt.title('Heatmap of Heston PINN solution at t=0')
plt.tight_layout()
plt.show()

# Training loss plot
plt.figure(figsize=(8,5))
plt.semilogy(np.arange(0, len(loss_hist))*print_every, loss_hist)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PINN training loss (Heston)')
plt.grid(True)
plt.tight_layout()
plt.show()
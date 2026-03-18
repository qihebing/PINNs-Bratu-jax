import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
import ml_collections
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, vmap
from flax.training import train_state
from functools import partial

# ----- Neural Network ----- #
class MLP(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)

# ----------------------------
# Analytical Solution (for N=1)
# ----------------------------
def analytical_solution_bratu(x, lam):
    from scipy.optimize import fsolve
    from numpy import sqrt, cosh, log

    def theta_eq(theta):
        return theta - sqrt(2 * lam) * cosh(theta / 4)

    theta = fsolve(theta_eq, 1.0)[0]
    return -2 * jnp.log(cosh((x - 0.5) * theta / 2) / cosh(theta / 4))

# ----------------------------
# PINN residual function
# ----------------------------
def net_u(params, model, x):
    return model.apply(params, x.reshape(-1, 1)).squeeze()

def pde_residual(params, model, x, N, lam):
    u_fn = lambda x_: net_u(params, model, x_)
    u = u_fn(x)
    u_x = grad(u_fn)(x)
    u_xx = grad(grad(u_fn))(x)
    return u_xx + (N - 1) / x * u_x + lam * jnp.exp(u)

# ----------------------------
# Loss function
# ----------------------------
def loss_fn(params, model, x_r, x_b, N, lam):
    res = vmap(lambda x: pde_residual(params, model, x, N, lam))(x_r)
    loss_r = jnp.mean(res**2)
    u_b = net_u(params, model, x_b)
    loss_b = jnp.mean(u_b**2)
    return loss_r + loss_b, (loss_r, loss_b)

@partial(jax.jit, static_argnums=(1,))
def train_step(state, model, batch, N, lam):
    x_r, x_b = batch
    params = state.params
    (loss, (loss_r, loss_b)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, x_r, x_b, N, lam
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, loss_r, loss_b

# ----------------------------
# Config
# ----------------------------
def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.seed = 0
    cfg.hidden_dim = 32
    cfg.num_layers = 3
    cfg.learning_rate = 1e-3
    cfg.max_steps = 50000
    cfg.logging_interval = 5000
    cfg.num_points = 10000
    cfg.N_list = [1, 2, 3]
    cfg.lam_list = [1.0, 3.513830719, 5.0]
    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.project = "PINN-Bratu"
    return cfg

# ----------------------------
# Main training loop
# ----------------------------
def train_single_case(N, lam, cfg):
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(cfg.hidden_dim, cfg.num_layers)
    dummy_input = jnp.ones((1, 1))
    params = model.init(key, dummy_input)

    tx = optax.adam(cfg.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    x_r = jnp.linspace(0.01, 0.99, cfg.num_points)
    x_b = jnp.array([0.0, 1.0])
    batch = (x_r, x_b)

    run = wandb.init(project=cfg.wandb.project, name=f"N={N}_lambda={lam}", config=dict(cfg), reinit=True)
    for step in range(cfg.max_steps):
        state, loss, loss_r, loss_b = train_step(state, model, batch, N, lam)
        if step % cfg.logging_interval == 0:
            wandb.log({"loss": float(loss), "residual_loss": float(loss_r), "boundary_loss": float(loss_b)}, step=step)
            print(f"Step {step}: total = {loss:.3e}, res = {loss_r:.3e}, bc = {loss_b:.3e}")
    run.finish()

    return state.params, model

# ----------------------------
# Plotting
# ----------------------------
def plot_single_N1(params, model, lam):
    x = jnp.linspace(0, 1, 200)
    u_pred = vmap(lambda x_: net_u(params, model, x_))(x)
    u_true = analytical_solution_bratu(x, lam)
    error = jnp.abs(u_true - u_pred)

    plt.figure(figsize=(8, 5))
    plt.plot(x, u_pred, label="PINN")
    plt.plot(x, u_true, '--', label="True")
    plt.fill_between(x, error, alpha=0.3, label="|Error|")
    plt.title(f"N=1, lambda={lam:.3f}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_N_family(results, N):
    x = jnp.linspace(0, 1, 200)
    plt.figure(figsize=(8, 5))
    for lam, (params, model) in results.items():
        u_pred = vmap(lambda x_: net_u(params, model, x_))(x)
        plt.plot(x, u_pred, label=f"lambda={lam:.3f}")
    plt.title(f"PINN solutions for N={N}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_combos(all_results):
    x = jnp.linspace(0, 1, 200)
    plt.figure(figsize=(10, 6))
    for (N, lam), (params, model) in all_results.items():
        u_pred = vmap(lambda x_: net_u(params, model, x_))(x)
        plt.plot(x, u_pred, label=f"N={N}, lambda={lam:.3f}")
    plt.title("All PINN Solutions")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    cfg = get_config()
    all_results = {}
    for N in cfg.N_list:
        N_results = {}
        for lam in cfg.lam_list:
            params, model = train_single_case(N, lam, cfg)
            all_results[(N, lam)] = (params, model)
            N_results[lam] = (params, model)
            if N == 1:
                plot_single_N1(params, model, lam)
        plot_N_family(N_results, N)
    plot_all_combos(all_results)

if __name__ == "__main__":
    main()

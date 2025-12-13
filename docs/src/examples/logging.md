# Logging & Experiment Tracking

DRiL.jl supports two popular logging backends for tracking your experiments:

- **TensorBoard** - Local logging with web-based visualization
- **Weights & Biases (W&B)** - Cloud-based experiment tracking with team collaboration features

Both backends automatically log training metrics during `train!()` calls, including:
- Episode rewards and lengths
- Policy/value/entropy losses
- Learning rate, gradient norms
- Frames per second (FPS)

## TensorBoard Example

TensorBoard is great for local development and quick experiments. Logs are stored locally and viewed through a web interface.

### Setup

```julia
using Pkg
Pkg.add("TensorBoardLogger")
```

### Example

```julia
using DRiL
using ClassicControlEnvironments
using TensorBoardLogger
using Logging
using Random
using Zygote

# Configuration
LOG_DIR = joinpath(pwd(), "data", "tensorboard")
N_ENVS = 4
TOTAL_TIMESTEPS = 50_000
SEED = 42

# Hyperparameter configs to compare
configs = [
    (name = "default", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.01f0),
]

# Run experiments
for config in configs
    # Create parallel environments
    rng = Random.Xoshiro(SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    DRiL.reset!(env)

    # Create PPO algorithm
    alg = PPO(;
        learning_rate = config.learning_rate,
        n_steps = config.n_steps,
        batch_size = config.batch_size,
        epochs = config.epochs,
        ent_coef = config.ent_coef,
    )

    # Create TensorBoard logger - each run gets its own subdirectory
    run_dir = joinpath(LOG_DIR, "CartPole_$(config.name)")
    tb_logger = TBLogger(run_dir; min_level = Logging.Info)

    # Create agent with logger
    layer = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(layer, alg; logger = tb_logger, verbose = 1, rng = rng)

    # Log hyperparameters (viewable in TensorBoard HPARAMS tab)
    log_hparams!(
        agent.logger,
        Dict(String(k) => v for (k, v) in pairs(config)),
        ["env/ep_rew_mean", "train/loss"],
    )

    # Train - metrics auto-logged to TensorBoard
    @info "Training" config = config.name log_dir = run_dir
    train!(agent, env, alg, TOTAL_TIMESTEPS)

    # Close logger
    close!(agent.logger)
end
```

### Viewing Results

Run in terminal:
```bash
tensorboard --logdir=data/tensorboard
```

Then open your browser at [http://localhost:6006](http://localhost:6006)

- **SCALARS** tab: Training metrics over time
- **HPARAMS** tab: Compare hyperparameter configurations

---

## Weights & Biases Example

W&B is ideal for cloud-based tracking, team collaboration, and hyperparameter sweeps.

### Setup

1. Create account at [wandb.ai](https://wandb.ai/site)
2. Install and authenticate:

```julia
using Pkg
Pkg.add("Wandb")

using Wandb
Wandb.login()  # Enter API key from https://wandb.ai/authorize
```

### Example

```julia
using DRiL
using ClassicControlEnvironments
using Wandb
using Random
using Zygote

# Configuration
WANDB_PROJECT = "DRiL-Examples"
N_ENVS = 4
TOTAL_TIMESTEPS = 50_000
SEED = 42

# Hyperparameter configs to compare
configs = [
    (name = "default", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "high_lr", learning_rate = 1.0f-3, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.0f0),
    (name = "with_entropy", learning_rate = 3.0f-4, n_steps = 128, batch_size = 64, epochs = 4, ent_coef = 0.01f0),
]

# Run experiments
for config in configs
    # Create parallel environments
    rng = Random.Xoshiro(SEED)
    envs = [CartPoleEnv(; rng = Random.Xoshiro(SEED + i)) for i in 1:N_ENVS]
    env = MonitorWrapperEnv(BroadcastedParallelEnv(envs))
    reset!(env)

    # Create PPO algorithm
    alg = PPO(;
        learning_rate = config.learning_rate,
        n_steps = config.n_steps,
        batch_size = config.batch_size,
        epochs = config.epochs,
        ent_coef = config.ent_coef,
    )

    # Create W&B logger - each run appears as separate experiment in dashboard
    wb_logger = WandbLogger(;
        project = WANDB_PROJECT,
        name = "CartPole_$(config.name)",
        config = Dict(String(k) => v for (k, v) in pairs(config)),
    )

    # Create agent with logger
    layer = ActorCriticLayer(observation_space(env), action_space(env))
    agent = Agent(layer, alg; logger = wb_logger, verbose = 1, rng = rng)

    # Train - metrics auto-logged to W&B
    @info "Training" config = config.name
    train!(agent, env, alg, TOTAL_TIMESTEPS)

    # Always close logger to finalize the W&B run
    close!(agent.logger)
end
```

### Viewing Results

After training, view results at `https://wandb.ai/<username>/DRiL-Examples`

- Compare runs side-by-side
- Use parallel coordinates for hyperparameter analysis
- Share results with team members

---

## Custom Logging

You can log additional metrics using the logging interface. The logger is accessible via `agent.logger`.

### Available Functions

```julia
# Log a single scalar value
log_scalar!(agent.logger, "custom/my_metric", value)

# Log multiple values at once
log_dict!(agent.logger, Dict(
    "custom/metric1" => value1,
    "custom/metric2" => value2,
))

# Log hyperparameters (typically done once at start)
log_hparams!(agent.logger, hparams_dict, metric_names)

# Manually set or increment the step counter
set_step!(agent.logger, step)
increment_step!(agent.logger, delta)
```

### Example: Logging Custom Metrics in a Callback

```julia
using DRiL

# Define a custom callback that logs additional metrics
struct CustomMetricsCallback <: AbstractCallback end

function DRiL.on_rollout_end(cb::CustomMetricsCallback, locals::Dict)
    agent = locals["agent"]
    env = locals["env"]
    
    # Log custom metrics
    log_scalar!(agent.logger, "custom/episode_count", length(env.episode_stats.episode_returns))
    
    # Log environment-specific info
    if !isempty(env.episode_stats.episode_returns)
        log_scalar!(agent.logger, "custom/best_return", maximum(env.episode_stats.episode_returns))
    end
    
    return true  # Continue training
end

# Use the callback during training
callbacks = [CustomMetricsCallback()]
train!(agent, env, alg, total_steps; callbacks = callbacks)
```

### No Logger (Default)

By default, agents use `NoTrainingLogger()` which discards all log calls. You don't need to pass anything to disable logging:

```julia
# Logging is disabled by default
agent = Agent(layer, alg)

# Equivalent to:
agent = Agent(layer, alg; logger = NoTrainingLogger())
```
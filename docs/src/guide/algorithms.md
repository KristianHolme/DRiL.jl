# Algorithms

::: tip Required Imports
All examples assume `using DRiL, Zygote` (Zygote is required for automatic differentiation).
:::

## PPO (Proximal Policy Optimization)

On-policy algorithm using clipped surrogate objective.

```julia
ppo = PPO(;
    gamma = 0.99f0,
    gae_lambda = 0.95f0,
    clip_range = 0.2f0,
    ent_coef = 0.0f0,
    vf_coef = 0.5f0,
    max_grad_norm = 0.5f0,
    n_steps = 2048,
    batch_size = 64,
    epochs = 10,
    learning_rate = 3f-4,
)

model = ActorCriticLayer(obs_space, act_space)
agent = Agent(model, ppo)
train!(agent, env, ppo, max_steps)
```

**Best for:** Discrete actions, robotic control, games.

## SAC (Soft Actor-Critic)

Off-policy algorithm with entropy regularization and twin Q-networks.

```julia
sac = SAC(;
    learning_rate = 3f-4,
    buffer_capacity = 1_000_000,
    batch_size = 256,
    tau = 0.005f0,
    gamma = 0.99f0,
    train_freq = 1,
    gradient_steps = 1,
    ent_coef = AutoEntropyCoefficient(),
)

model = SACPolicy(obs_space, act_space)
agent = Agent(model, sac)
train!(agent, env, sac, max_steps)
```

**Best for:** Continuous control, sample efficiency matters.

## Entropy Coefficient

```julia
# Fixed entropy
sac = SAC(ent_coef = FixedEntropyCoefficient(0.2f0))

# Automatic tuning (default)
sac = SAC(ent_coef = AutoEntropyCoefficient())
```


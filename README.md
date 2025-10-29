# DRiL.jl

[![Build Status](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

**Deep ReInforcement Learning** - A (aspirationally) high-performance Julia package for deep reinforcement learning algorithms.

## Overview

DRiL.jl is a prototype DRL package, aiming to be fast, flexible, and easy to use.

## Main Features

  
- **Modern Architecture**: Built on [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks with automatic differentiation support
- **Flexible Environments**: Comprehensive environment interface supporting both discrete and continuous action spaces
- **Rich Logging**: TensorBoard integration for training monitoring, and timer output ([TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl)) for performance analysis
- **Parallelization**: Built-in support for parallel environment execution

## Implemented Algorithms

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

## Core Components
The DRiL.jl package is built around the following core components: **Environments**, **Models**, **Agents**, and **Algorithms**.
The environment is the system we are interested in controlling, the model is the (actor–critic) neural network(s) used for control, the agent manages training, and the algorithm specifies the training procedure and loss.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KristianHolme/DRiL.jl")
```

## Quick Start Example

Here's a complete example training a PPO agent on the CartPole environment:

```julia
using DRiL
using Pkg
Pkg.add(url="https://github.com/KristianHolme/ClassicControlEnvironments.jl")
using ClassicControlEnvironments
using Random

## Environment
parallel_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:4])

## Actor-Critic Layer (training-time)
model = ActorCriticLayer(
    observation_space(parallel_env), 
    action_space(parallel_env)
)

## Algorithm
ppo = PPO(
    gamma=0.99f0,
    gae_lambda=0.95f0,
    clip_range=0.2f0,
    ent_coef=0.01f0,
    vf_coef=0.5f0,
    normalize_advantage=true
)

## Agent (simple unified constructor)
agent = Agent(model, ppo; verbose=2)

## Train
max_steps = 100_000
learn_stats, to = learn!(agent, parallel_env; max_steps)

## Evaluate the trained agent
eval_env = CartPoleEnv(max_steps=500)
eval_stats = evaluate_agent(agent, eval_env, n_episodes=10, deterministic=true)

println("Average episodic return: $(mean(eval_stats.episodic_returns))")
println("Average episode length: $(mean(eval_stats.episodic_lengths))")

# Print timer output
print_timer(to)
```

## Advanced Usage

### Custom Environments

Implement the DRiL environment interface:

```julia
struct MyEnv <: AbstractEnv
    # Your environment state
end

# Required methods
DRiL.reset!(env::MyEnv) = # Reset environment
DRiL.act!(env::MyEnv, action) = # Take action, return reward  
DRiL.observe(env::MyEnv) = # Return current observation
DRiL.terminated(env::MyEnv) = # Check if episode is done
DRiL.truncated(env::MyEnv) = # Check if episode is truncated
DRiL.action_space(env::MyEnv) = # Return action space
DRiL.observation_space(env::MyEnv) = # Return observation space
```

### Environment Wrappers

```julia
# Normalize observations and rewards
env = NormalizeWrapperEnv(env, normalize_obs=true, normalize_reward=true)

# Monitor episode statistics  
env = MonitorWrapperEnv(env)

# Scale observations and actions
env = ScalingWrapperEnv(env)
```

### Custom Layer Architectures

```julia
model = ActorCriticLayer(
    obs_space,
    act_space,
    hidden_dims=[128, 128, 64],  # Larger network
    activation=relu,              # Different activation
)
```

### Deployment (lightweight policy)

```julia
# Extract a deployment-time policy (actor-only)
dp = extract_policy(agent)

# Predict env-ready actions
env_actions = predict(dp, batch_of_obs; deterministic=true)
```

Notes:
- Action conversion (policy-space → env-space) is handled automatically via algorithm-selected adapters (no user code required).
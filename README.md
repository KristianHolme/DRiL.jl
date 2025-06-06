# DRiL.jl

[![Build Status](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/DRiL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

**Deep ReInforcement Learning** - A (aspirationally) high-performance Julia package for deep reinforcement learning algorithms.

## Overview

DRiL.jl is a prototype DRL package, aiming to be fast, flexible, and easy to use.

## Main Features

  
- **Modern Architecture**: Built on [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks with automatic differentiation support
- **Flexible Environments**: Comprehensive environment interface supporting both discrete and continuous action spaces
- **Rich Logging**: TensorBoard integration for training monitoring
- **Parallelization**: Built-in support for parallel environment execution

## Implemented Algorithms

- PPO (Proximal Policy Optimization)

## Core Components
The DRiL.jl package is built around the following core components: **Environments**, **Policies**, **Agents**, and **Algorithms**.
The environment is the system we are interested in controlling, the policy is the neural network(s) that we use to control it, the agent manages the training of the policy, and the algorithm influences the many aspects of the training process, most notably the loss function.

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
Pkg.add("https://github.com/KristianHolme/ClassicControlEnvironments.jl")
using ClassicControlEnvironments
using Random

# Wrap environment for parallel execution (using 4 parallel environments)
parallel_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:4])

# Create a discrete actor-critic policy
policy = DiscreteActorCriticPolicy(
    observation_space(env), 
    action_space(env)
)

# Create an agent with the policy
agent = ActorCriticAgent(
    policy,
    n_steps=2048,        # Steps per rollout
    batch_size=64,       # Minibatch size
    epochs=10,          # Optimization epochs per update
    learning_rate=3f-4, # Learning rate
    verbose=2          # Enable progress bars and stats
)

# Configure PPO algorithm
ppo = PPO(
    gamma=0.99f0,         # Discount factor
    gae_lambda=0.95f0,    # GAE parameter
    clip_range=0.2f0,     # PPO clipping parameter
    ent_coef=0.01f0,      # Entropy bonus coefficient
    vf_coef=0.5f0,        # Value function loss coefficient
    normalize_advantage=true
)

# Train the agent
max_steps = 100_000
learn_stats = learn!(agent, parallel_env, ppo; max_steps)

# Evaluate the trained agent
eval_env = CartPoleEnv(max_steps=500)
eval_stats = evaluate_agent(agent, eval_env, n_episodes=10, deterministic=true)

println("Average episodic return: $(mean(eval_stats.episodic_returns))")
println("Average episode length: $(mean(eval_stats.episodic_lengths))")
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

### Custom Network Architectures

```julia
# Custom policy with different hidden dimensions
policy = DiscreteActorCriticPolicy(
    obs_space,
    act_space, 
    hidden_dims=[128, 128, 64],  # Larger network
    activation=relu,              # Different activation
)
```
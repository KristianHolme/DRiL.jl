# DRiL.jl Documentation

Welcome to the documentation for DRiL.jl!

DRiL is a Julia package for deep reinforcement learning.

Core concepts:
- Environments: your task/problem, with `AbstractEnv` interface
- Models: training-time actor-critic models (actor + critic(s))
- Algorithms: PPO, SAC (more to come)
- Agents: orchestrate training for a model with a given algorithm

## Quick Start

```julia
using DRiL
using ClassicControlEnvironments

# Parallel envs
env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:4])

# Model
model = DiscreteActorCriticPolicy(observation_space(env), action_space(env))

# Algorithm
alg = PPO()

# Agent
agent = Agent(model, alg; verbose=1)

# Train
learn!(agent, env; max_steps=50_000)

# Deployment policy
dp = extract_policy(agent)
actions = predict(dp, batch_of_observations; deterministic=true)
```


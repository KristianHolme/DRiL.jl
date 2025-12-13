
# Environments {#Environments}

## Interface {#Interface}

Implement `AbstractEnv` with these required methods:

```julia
struct MyEnv <: AbstractEnv
    # state fields
end

DRiL.reset!(env::MyEnv)           # Reset to initial state
DRiL.act!(env::MyEnv, action)     # Take action, return reward
DRiL.observe(env::MyEnv)          # Return current observation
DRiL.terminated(env::MyEnv)       # Terminal state reached?
DRiL.truncated(env::MyEnv)        # Time limit reached?
DRiL.action_space(env::MyEnv)     # Return action space
DRiL.observation_space(env::MyEnv) # Return observation space
```


## Spaces {#Spaces}

```julia
# Continuous actions/observations
Box{Float32}(low, high)  # low/high are vectors

# Discrete actions
Discrete(n)              # Actions: 1, 2, ..., n
Discrete(n, start=0)     # Actions: 0, 1, ..., n-1
```


## Parallel Environments {#Parallel-Environments}

```julia
# Multi-threaded (multi-threaded, keep threading overhead in mind!)
env = MultiThreadedParallelEnv([MyEnv() for _ in 1:n_envs])

# Broadcasted (single-threaded, often fastest for cheap environments like those in ClassicControlEnvironments.jl)
env = BroadcastedParallelEnv([MyEnv() for _ in 1:n_envs])
```


## Environment Validation {#Environment-Validation}

```julia
check_env(env)  # Validates interface implementation
```


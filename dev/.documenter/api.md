
# API Reference {#API-Reference}

## Environments {#Environments}
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.AbstractEnv' href='#DRiL.AbstractEnv'><span class="jlbinding">DRiL.AbstractEnv</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractEnv
```


Abstract base type for all reinforcement learning environments.

Subtypes must implement the following methods:
- `reset!(env)` - Reset the environment
  
- `act!(env, action)` - Take an action and return the reward
  
- `observe(env)` - Get current observation
  
- `terminated(env)` - Check if episode terminated
  
- `truncated(env)` - Check if episode was truncated
  
- `action_space(env)` - Get the action space
  
- `observation_space(env)` - Get the observation space
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L5-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.AbstractParallelEnv' href='#DRiL.AbstractParallelEnv'><span class="jlbinding">DRiL.AbstractParallelEnv</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractParallelEnv <: AbstractEnv
```


Abstract type for vectorized/parallel environments that manage multiple environment instances.

**Key Differences from AbstractEnv**

|       Method |              Single Env |                                        Parallel Env |
| ------------:| -----------------------:| ---------------------------------------------------:|
|    `observe` | Returns one observation |                      Returns vector of observations |
|       `act!` |        Returns `reward` | Returns `(rewards, terminateds, truncateds, infos)` |
| `terminated` |          Returns `Bool` |                              Returns `Vector{Bool}` |
|  `truncated` |          Returns `Bool` |                              Returns `Vector{Bool}` |


**Auto-Reset Behavior**

Parallel environments automatically reset individual sub-environments when they terminate or truncate. The terminal observation is stored in `infos[i]["terminal_observation"]` before reset.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L21-L38" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.reset!' href='#DRiL.reset!'><span class="jlbinding">DRiL.reset!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
reset!(env::AbstractEnv) -> Nothing
```


Reset the environment to its initial state.

**Arguments**
- `env::AbstractEnv`: The environment to reset
  

**Returns**
- `Nothing`
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L41-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.act!' href='#DRiL.act!'><span class="jlbinding">DRiL.act!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
act!(env::AbstractEnv, action) -> reward
```


Take an action in the environment and return the reward.

**Arguments**
- `env::AbstractEnv`: The environment to act in
  
- `action`: The action to take (type depends on environment&#39;s action space)
  

**Returns**
- `reward`: Numerical reward from taking the action
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L54-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.observe' href='#DRiL.observe'><span class="jlbinding">DRiL.observe</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
observe(env::AbstractEnv) -> observation
```


Get the current observation from the environment.

**Arguments**
- `env::AbstractEnv`: The environment to observe
  

**Returns**
- `observation`: Current state observation (type/shape depends on environment&#39;s observation space)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L68-L78" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.terminated' href='#DRiL.terminated'><span class="jlbinding">DRiL.terminated</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
terminated(env::AbstractEnv) -> Bool
```


Check if the environment episode has terminated due to reaching a terminal state.

**Arguments**
- `env::AbstractEnv`: The environment to check
  

**Returns**
- `Bool`: `true` if episode is terminated, `false` otherwise
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L81-L91" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.truncated' href='#DRiL.truncated'><span class="jlbinding">DRiL.truncated</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
truncated(env::AbstractEnv) -> Bool
```


Check if the environment episode has been truncated (e.g., time limit reached).

**Arguments**
- `env::AbstractEnv`: The environment to check
  

**Returns**
- `Bool`: `true` if episode is truncated, `false` otherwise
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L94-L104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.action_space' href='#DRiL.action_space'><span class="jlbinding">DRiL.action_space</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
action_space(env::AbstractEnv) -> AbstractSpace
```


Get the action space specification for the environment.

**Arguments**
- `env::AbstractEnv`: The environment
  

**Returns**
- `AbstractSpace`: The action space (e.g., Box, Discrete)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L107-L117" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.observation_space' href='#DRiL.observation_space'><span class="jlbinding">DRiL.observation_space</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
observation_space(env::AbstractEnv) -> AbstractSpace
```


Get the observation space specification for the environment.

**Arguments**
- `env::AbstractEnv`: The environment
  

**Returns**
- `AbstractSpace`: The observation space (e.g., Box, Discrete)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L120-L130" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.get_info' href='#DRiL.get_info'><span class="jlbinding">DRiL.get_info</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_info(env::AbstractEnv) -> Dict
```


Get additional environment information (metadata, debug info, etc.).

**Arguments**
- `env::AbstractEnv`: The environment
  

**Returns**
- `Dict`: Dictionary containing environment-specific information
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L133-L143" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.number_of_envs' href='#DRiL.number_of_envs'><span class="jlbinding">DRiL.number_of_envs</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
number_of_envs(env::AbstractParallelEnv) -> Int
```


Get the number of parallel environments in a parallel environment wrapper.

**Arguments**
- `env::AbstractParallelEnv`: The parallel environment
  

**Returns**
- `Int`: Number of parallel environments
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/environments.jl#L146-L156" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Spaces {#Spaces}
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.AbstractSpace' href='#DRiL.AbstractSpace'><span class="jlbinding">DRiL.AbstractSpace</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractSpace
```


Abstract base type for all observation and action spaces in DRiL.jl. Concrete subtypes include `Box` (continuous) and `Discrete` (finite actions).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/spaces.jl#L1-L6" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.Box' href='#DRiL.Box'><span class="jlbinding">DRiL.Box</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Box{T <: Number} <: AbstractSpace
```


A continuous space with lower and upper bounds per dimension.

**Fields**
- `low::Array{T}`: Lower bounds for each dimension
  
- `high::Array{T}`: Upper bounds for each dimension  
  
- `shape::Tuple{Vararg{Int}}`: Shape of the space
  

**Example**

```julia
# 2D box with different bounds per dimension
space = Box(Float32[-1, -2], Float32[1, 3])

# Uniform bounds
space = Box(-1.0f0, 1.0f0, (4,))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/spaces.jl#L9-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.Discrete' href='#DRiL.Discrete'><span class="jlbinding">DRiL.Discrete</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Discrete{T <: Integer} <: AbstractSpace
```


A discrete space representing a finite set of actions.

**Fields**
- `n::T`: Number of discrete actions
  
- `start::T`: Starting index (default: 1 for Julia convention)
  

**Example**

```julia
space = Discrete(4)  # Actions: 1, 2, 3, 4
space = Discrete(4, 0)  # Actions: 0, 1, 2, 3 (Gym convention)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/spaces.jl#L141-L155" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Algorithms {#Algorithms}
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.PPO' href='#DRiL.PPO'><span class="jlbinding">DRiL.PPO</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PPO{T <: AbstractFloat} <: OnPolicyAlgorithm
```


Proximal Policy Optimization algorithm.

**Fields**
- `gamma`: Discount factor (default: 0.99)
  
- `gae_lambda`: GAE lambda for advantage estimation (default: 0.95)
  
- `clip_range`: PPO clipping parameter (default: 0.2)
  
- `ent_coef`: Entropy coefficient (default: 0.0)
  
- `vf_coef`: Value function coefficient (default: 0.5)
  
- `max_grad_norm`: Maximum gradient norm for clipping (default: 0.5)
  
- `n_steps`: Steps per rollout before update (default: 2048)
  
- `batch_size`: Minibatch size (default: 64)
  
- `epochs`: Number of epochs per update (default: 10)
  
- `learning_rate`: Optimizer learning rate (default: 3e-4)
  

**Example**

```julia
ppo = PPO(gamma=0.99f0, n_steps=2048, epochs=10)
agent = Agent(model, ppo)
train!(agent, env, ppo, 100_000)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/algorithms/ppo.jl#L1-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.SAC' href='#DRiL.SAC'><span class="jlbinding">DRiL.SAC</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SAC{T <: AbstractFloat, E <: AbstractEntropyCoefficient} <: OffPolicyAlgorithm
```


Soft Actor-Critic algorithm with automatic entropy tuning.

**Fields**
- `learning_rate`: Optimizer learning rate (default: 3e-4)
  
- `buffer_capacity`: Replay buffer size (default: 1M)
  
- `start_steps`: Random exploration steps before training (default: 100)
  
- `batch_size`: Batch size for updates (default: 256)
  
- `tau`: Soft update coefficient for target networks (default: 0.005)
  
- `gamma`: Discount factor (default: 0.99)
  
- `train_freq`: Steps between gradient updates (default: 1)
  
- `gradient_steps`: Gradient steps per update, -1 for auto (default: 1)
  
- `ent_coef`: Entropy coefficient (`AutoEntropyCoefficient` or `FixedEntropyCoefficient`)
  

**Example**

```julia
sac = SAC(learning_rate=3f-4, buffer_capacity=1_000_000)
model = SACPolicy(obs_space, act_space)
agent = Agent(model, sac)
train!(agent, env, sac, 500_000)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/algorithms/sac.jl#L1-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `train!`. Check Documenter&#39;s build log for details.

:::

## Agents {#Agents}
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.Agent' href='#DRiL.Agent'><span class="jlbinding">DRiL.Agent</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Unified Agent for all algorithms.

verbose:     0: nothing     1: progress bar     2: progress bar and stats


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/agents/agent_types.jl#L44-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.predict_actions' href='#DRiL.predict_actions'><span class="jlbinding">DRiL.predict_actions</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
predict_actions(layer::AbstractLayer, obs::AbstractArray, ps, st; deterministic::Bool=false) -> (actions, st)
```


Predict actions from batched observations.

**Arguments**
- `layer::AbstractLayer`: The actor-critic layer
  
- `obs::AbstractArray`: Batched observations (last dimension is batch)
  
- `ps`: Layer parameters
  
- `st`: Layer state
  
- `deterministic::Bool=false`: Whether to use deterministic actions
  

**Returns**
- `actions`: Vector/Array of actions (raw layer outputs, not processed for environment)
  
- `st`: Updated layer state
  

**Notes**
- Input observations must be batched (matrix/array format)
  
- Output actions are raw layer outputs (e.g., 1-based for Discrete layers)
  
- Use `to_env()` to convert for environment use
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/layers.jl#L11-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.predict_values' href='#DRiL.predict_values'><span class="jlbinding">DRiL.predict_values</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
predict_values(layer::AbstractLayer, obs::AbstractArray, [actions::AbstractArray,] ps, st) -> (values, st)
```


Predict Q-values from batched observations and actions (for Q-Critic layers).

**Arguments**
- `layer::AbstractLayer`: The actor-critic layer
  
- `obs::AbstractArray`: Batched observations (last dimension is batch)
  
- `actions::AbstractArray`: Batched actions (last dimension is batch) (only for Q-Critic layers)
  
- `ps`: Layer parameters
  
- `st`: Layer state
  

**Returns**
- `values`: batched values (tuples of values for multiple Q-Critic networks)
  
- `st`: Updated layer state
  

**Notes**
- Input observations and actions must be batched (matrix/array format)
  
- Actions should be in raw layer format (e.g., 1-based for Discrete)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/layers.jl#L35-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `steps_taken`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.evaluate_agent' href='#DRiL.evaluate_agent'><span class="jlbinding">DRiL.evaluate_agent</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
evaluate_agent(agent, env; kwargs...)
```


Evaluate a policy/agent for a specified number of episodes and return performance statistics.

**Arguments**
- `agent`: The agent to evaluate (must implement `predict` method)
  
- `env`: The environment to evaluate on (single env or parallel env)
  

**Keyword Arguments**
- `n_eval_episodes::Int = 10`: Number of episodes to evaluate
  
- `deterministic::Bool = true`: Whether to use deterministic actions
  
- `render::Bool = false`: Whether to render the environment
  
- `callback::Union{Nothing, Function} = nothing`: Optional callback function called after each step
  
- `reward_threshold::Union{Nothing, Real} = nothing`: Minimum expected mean reward (throws error if not met)
  
- `return_episode_rewards::Bool = false`: If true, returns individual episode rewards and lengths
  
- `warn::Bool = true`: Whether to warn about missing Monitor wrapper
  
- `rng::AbstractRNG = Random.default_rng()`: Random number generator for reproducible evaluation
  

**Returns**
- If `return_episode_rewards = false`: `(mean_reward::Float64, std_reward::Float64)`
  
- If `return_episode_rewards = true`: `(episode_rewards::Vector{Float64}, episode_lengths::Vector{Int})`
  

**Notes**
- Episodes are distributed evenly across parallel environments to remove bias
  
- If environment is wrapped with Monitor, episode statistics from Monitor are used
  
- Otherwise, rewards and lengths are tracked manually during evaluation
  
- For environments with reward/length modifying wrappers, consider using Monitor wrapper
  

**Examples**

```julia
# Basic evaluation
mean_reward, std_reward = evaluate_agent(agent, env; n_eval_episodes=20)

# Get individual episode data
episode_rewards, episode_lengths = evaluate_agent(agent, env; 
    return_episode_rewards=true, deterministic=false)

# Evaluation with threshold check
mean_reward, std_reward = evaluate_agent(agent, env; 
    reward_threshold=100.0, n_eval_episodes=50)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/evaluation.jl#L11-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Layers {#Layers}

::: warning Missing docstring.

Missing docstring for `ActorCriticLayer`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `ContinuousActorCriticLayer`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `DiscreteActorCriticLayer`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `SACPolicy`. Check Documenter&#39;s build log for details.

:::

## Buffers {#Buffers}

::: warning Missing docstring.

Missing docstring for `RolloutBuffer`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.ReplayBuffer' href='#DRiL.ReplayBuffer'><span class="jlbinding">DRiL.ReplayBuffer</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ReplayBuffer{T,O,OBS,AC}
```


A circular buffer for storing multiple trajectories of off-policy experience data, used for replay-based learning algorithms.

**Truncation Logic**
- If `terminated = true`, then there should be no `truncated_observation`
  
- If `truncated = true`, then there should be a `truncated_observation`  
  
- If `terminated = false` and `truncated = false`, then we stopped in the middle of an episode, so there should be a `truncated_observation`
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/buffers/buffer_types.jl#L42-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Wrappers {#Wrappers}

::: warning Missing docstring.

Missing docstring for `MultiThreadedParallelEnv`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `BroadcastedParallelEnv`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `NormalizeWrapperEnv`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `ScalingWrapperEnv`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `MonitorWrapperEnv`. Check Documenter&#39;s build log for details.

:::

## Deployment {#Deployment}
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.extract_policy' href='#DRiL.extract_policy'><span class="jlbinding">DRiL.extract_policy</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
extract_policy(agent) -> DeploymentPolicy
```


Create a lightweight deployment policy from a trained agent.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/deployment/deployment_policy.jl#L11-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `DeploymentPolicy`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `NormalizedDeploymentPolicy`. Check Documenter&#39;s build log for details.

:::

## Logging {#Logging}

::: warning Missing docstring.

Missing docstring for `AbstractTrainingLogger`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `NoTrainingLogger`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='DRiL.log_scalar!' href='#DRiL.log_scalar!'><span class="jlbinding">DRiL.log_scalar!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
log_scalar!(logger::AbstractTrainingLogger, key::AbstractString, value::Real)
```


Log a single scalar metric under `key`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/logging.jl#L14-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.log_dict!' href='#DRiL.log_dict!'><span class="jlbinding">DRiL.log_dict!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
log_dict!(logger::AbstractTrainingLogger, kv::AbstractDict{<:AbstractString,<:Any})
```


Log multiple metrics at once from a string-keyed dictionary.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/logging.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='DRiL.set_step!' href='#DRiL.set_step!'><span class="jlbinding">DRiL.set_step!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
set_step!(logger::AbstractTrainingLogger, step::Integer)
```


Set the global step for subsequent metric logs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KristianHolme/DRiL.jl/blob/da55e64aad5bc8ede59786d4c958ab75770c832b/src/interfaces/logging.jl#L6-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Callbacks {#Callbacks}

::: warning Missing docstring.

Missing docstring for `AbstractCallback`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `on_training_start`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `on_training_end`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `on_rollout_start`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `on_rollout_end`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `on_step`. Check Documenter&#39;s build log for details.

:::

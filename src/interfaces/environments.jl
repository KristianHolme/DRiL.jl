# ------------------------------------------------------------
# Environments
# ------------------------------------------------------------
abstract type AbstractEnv end
abstract type AbstractParallelEnv <: AbstractEnv end

"""
    reset!(env::AbstractEnv) -> Nothing

Reset the environment to its initial state.

# Arguments
- `env::AbstractEnv`: The environment to reset

# Returns
- `Nothing`
"""
function reset! end

"""
    act!(env::AbstractEnv, action) -> reward

Take an action in the environment and return the reward.

# Arguments
- `env::AbstractEnv`: The environment to act in
- `action`: The action to take (type depends on environment's action space)

# Returns
- `reward`: Numerical reward from taking the action
"""
function act! end

"""
    observe(env::AbstractEnv) -> observation

Get the current observation from the environment.

# Arguments
- `env::AbstractEnv`: The environment to observe

# Returns
- `observation`: Current state observation (type/shape depends on environment's observation space)
"""
function observe end

"""
    terminated(env::AbstractEnv) -> Bool

Check if the environment episode has terminated due to reaching a terminal state.

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if episode is terminated, `false` otherwise
"""
function terminated end

"""
    truncated(env::AbstractEnv) -> Bool

Check if the environment episode has been truncated (e.g., time limit reached).

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if episode is truncated, `false` otherwise
"""
function truncated end

"""
    action_space(env::AbstractEnv) -> AbstractSpace

Get the action space specification for the environment.

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `AbstractSpace`: The action space (e.g., Box, Discrete)
"""
function action_space end

"""
    observation_space(env::AbstractEnv) -> AbstractSpace

Get the observation space specification for the environment.

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `AbstractSpace`: The observation space (e.g., Box, Discrete)
"""
function observation_space end

"""
    get_info(env::AbstractEnv) -> Dict

Get additional environment information (metadata, debug info, etc.).

# Arguments
- `env::AbstractEnv`: The environment

# Returns
- `Dict`: Dictionary containing environment-specific information
"""
function get_info end

"""
    number_of_envs(env::AbstractParallelEnv) -> Int

Get the number of parallel environments in a parallel environment wrapper.

# Arguments
- `env::AbstractParallelEnv`: The parallel environment

# Returns
- `Int`: Number of parallel environments
"""
function number_of_envs end


# ------------------------------------------------------------
# Environment wrappers
# ------------------------------------------------------------
abstract type AbstractEnvWrapper{E <: AbstractEnv} <: AbstractEnv end
abstract type AbstractParallelEnvWrapper{E <: AbstractParallelEnv} <: AbstractParallelEnv end


# ------------------------------------------------------------
# Environment wrapper utilities
# ------------------------------------------------------------
"""
    is_wrapper(env::AbstractEnv) -> Bool

Check if an environment is a wrapper around another environment.

# Arguments
- `env::AbstractEnv`: The environment to check

# Returns
- `Bool`: `true` if environment is a wrapper, `false` otherwise
"""
is_wrapper(env::AbstractEnv) = env isa AbstractEnvWrapper
is_wrapper(env::AbstractParallelEnv) = env isa AbstractParallelEnvWrapper

"""
    unwrap(env::AbstractEnvWrapper) -> AbstractEnv

Unwrap one layer of environment wrapper to access the underlying environment.

# Arguments
- `env::AbstractEnvWrapper`: The wrapped environment

# Returns
- `AbstractEnv`: The underlying environment (may still be wrapped)
"""
function unwrap end


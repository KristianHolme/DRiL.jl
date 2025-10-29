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
# Agents
# ------------------------------------------------------------
abstract type AbstractAgent end

abstract type AbstractBuffer end

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

# ------------------------------------------------------------
# Logging interface
# ------------------------------------------------------------
abstract type AbstractTrainingLogger end

"""
    set_step!(logger::AbstractTrainingLogger, step::Integer)

Set the global step for subsequent metric logs.
"""
function set_step! end
function increment_step! end

"""
    log_scalar!(logger::AbstractTrainingLogger, key::AbstractString, value::Real)

Log a single scalar metric under `key`.
"""
function log_scalar! end

"""
    log_dict!(logger::AbstractTrainingLogger, kv::AbstractDict{<:AbstractString,<:Any})

Log multiple metrics at once from a string-keyed dictionary.
"""
function log_dict! end

"""
    write_hparams!(logger::AbstractTrainingLogger, hparams::AbstractDict{<:AbstractString,<:Any}, metrics::AbstractVector{<:AbstractString})

Write hyperparameters and associate them with specified metrics for hyperparameter tuning.
"""
function write_hparams! end

"""
    flush!(logger::AbstractTrainingLogger)

Ensure any buffered data is pushed to the backend. Implementations may no-op.
"""
function flush! end

"""
    close!(logger::AbstractTrainingLogger)

Finalize the logger and release resources. Implementations may no-op.
"""
function close! end

# Strict conversion gate: identity for wrapper, error otherwise
Base.convert(::Type{AbstractTrainingLogger}, x::AbstractTrainingLogger) = x
Base.convert(::Type{AbstractTrainingLogger}, x) = error(
    "Unsupported logger $(typeof(x)). Pass NoTrainingLogger(), TensorBoardLogger.TBLogger, or Wandb.WandbLogger."
)


"""
    log_stats(env::AbstractEnv, logger::AbstractLogger) -> Nothing

Log environment-specific statistics to a logger (optional interface).

# Arguments
- `env::AbstractEnv`: The environment
- `logger::AbstractLogger`: The logger to write to

# Returns
- `Nothing`

# Notes
Default implementation does nothing. Environments can override to log custom metrics.
"""
function log_stats(env::AbstractEnv, logger::AbstractTrainingLogger)
    return nothing
end

# ------------------------------------------------------------
# Algorithms
# ------------------------------------------------------------
abstract type AbstractAlgorithm end
abstract type OffPolicyAlgorithm <: AbstractAlgorithm end
abstract type OnPolicyAlgorithm <: AbstractAlgorithm end

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------
abstract type AbstractCallback end

# ------------------------------------------------------------
# Entropy target
# ------------------------------------------------------------
abstract type AbstractEntropyTarget end

struct FixedEntropyTarget{T <: AbstractFloat} <: AbstractEntropyTarget
    target::T
end

struct AutoEntropyTarget <: AbstractEntropyTarget end

# ------------------------------------------------------------
# Entropy coefficient
# ------------------------------------------------------------
abstract type AbstractEntropyCoefficient end

struct FixedEntropyCoefficient{T <: AbstractFloat} <: AbstractEntropyCoefficient
    coef::T
end

@kwdef struct AutoEntropyCoefficient{T <: AbstractFloat, E <: AbstractEntropyTarget} <: AbstractEntropyCoefficient
    target::E = AutoEntropyTarget()
    initial_value::T = 1.0f0
end

Base.string(e::AutoEntropyCoefficient) = "AutoEntropyCoefficient(target=$(e.target), initial_value=$(e.initial_value))"
Base.string(e::FixedEntropyCoefficient) = "FixedEntropyCoefficient(coef=$(e.coef))"

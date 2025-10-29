# ------------------------------------------------------------
# Policies
# ------------------------------------------------------------
abstract type AbstractPolicy <: Lux.AbstractLuxLayer end
abstract type CriticType end
@kwdef struct QCritic <: CriticType
    n_critics::Int = 2
end
struct VCritic <: CriticType end

"""
    predict_actions(policy::AbstractPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false) -> (actions, st)

Predict actions from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `ps`: Policy parameters
- `st`: Policy state
- `deterministic::Bool=false`: Whether to use deterministic actions

# Returns
- `actions`: Vector/Array of actions (raw policy outputs, not processed for environment)
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
- Use `process_action()` to convert for environment use
"""
function predict_actions end


"""
    predict_values(policy::AbstractPolicy, obs::AbstractArray, [actions::AbstractArray,] ps, st) -> (values, st)

Predict Q-values from batched observations and actions (for Q-Critic policies).

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `actions::AbstractArray`: Batched actions (last dimension is batch) (only for Q-Critic policies)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `values`: batched values (tuples of values for multiple Q-Critic networks)
- `st`: Updated policy state

# Notes
- Input observations and actions must be batched (matrix/array format)
- Actions should be in raw policy format (e.g., 1-based for Discrete)
"""
function predict_values end


"""
    evaluate_actions(policy::AbstractPolicy, obs::AbstractArray, actions::AbstractArray, ps, st) -> (values, log_probs, entropy, st)

Evaluate given actions for batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `actions::AbstractArray`: Batched actions to evaluate (raw policy format)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities for the actions
- `entropy`: Vector of policy entropy values
- `st`: Updated policy state

# Notes
- All inputs must be batched (matrix/array format)
- Actions should be in raw policy format (e.g., 1-based for Discrete)
"""
function evaluate_actions end

"""
    action_log_prob(policy::AbstractPolicy, obs::AbstractArray, ps, st) -> (actions, log_probs, st)

Sample actions and return their log probabilities from batched observations (for SAC).

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `actions`: Vector/Array of sampled actions
- `log_probs`: Vector of log probabilities for the sampled actions
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
"""
function action_log_prob end

"""
    (policy::AbstractPolicy)(obs::AbstractArray, ps, st) -> (actions, values, log_probs, st)

Forward pass through policy: get actions, values, and log probabilities from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `actions`: Vector/Array of actions (raw policy outputs)
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
"""
function (policy::AbstractPolicy) end

abstract type AbstractNoise end

struct StateIndependantNoise <: AbstractNoise end
struct StateDependentNoise <: AbstractNoise end
struct NoNoise <: AbstractNoise end

abstract type AbstractActorCriticPolicy <: AbstractPolicy end

# Abstract types for shared features parameter
abstract type FeatureSharing end
struct SharedFeatures <: FeatureSharing end
struct SeparateFeatures <: FeatureSharing end


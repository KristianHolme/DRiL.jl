# Agent methods

#takes vector of observations
#TODO: adjust name?
"""
    get_action_and_values(agent::ActorCriticAgent, observations::AbstractVector) -> (actions, values, logprobs)

Get actions, values, and log probabilities for a vector of observations.

# Arguments
- `agent::ActorCriticAgent`: The agent
- `observations::AbstractVector`: Vector of observations

# Returns  
- `actions`: Vector of actions (processed for environment use)
- `values`: Vector of value estimates
- `logprobs`: Vector of log probabilities
"""
function get_action_and_values(agent::ActorCriticAgent, observations::AbstractVector)
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(layer))
    #FIXME: type instability here, is policy not known??
    actions, values, logprobs, st = layer(batched_obs, ps, st)
    @reset train_state.states = st
    agent.train_state = train_state
    return actions, values, logprobs
end

"""
    predict_values(agent::ActorCriticAgent, observations::AbstractVector) -> Vector

Predict value estimates for a vector of observations.

# Arguments
- `agent::ActorCriticAgent`: The agent
- `observations::AbstractVector`: Vector of observations

# Returns
- `Vector`: Value estimates for each observation
"""
function predict_values(agent::ActorCriticAgent, observations::AbstractVector)
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(layer))
    values, st = predict_values(layer, batched_obs, ps, st)
    @reset train_state.states = st
    agent.train_state = train_state
    return values
end

"""
    predict_actions(agent::ActorCriticAgent, observations::AbstractVector; kwargs...) -> Vector

Predict actions for a vector of observations, processed for environment use.

# Arguments
- `agent::ActorCriticAgent`: The agent
- `observations::AbstractVector`: Vector of observations
- `deterministic::Bool=false`: Whether to use deterministic actions
- `rng::AbstractRNG=agent.rng`: Random number generator
- `raw::Bool=false`: Whether to return raw actions (not processed for environment). Not supported for ActorCriticAgent.

# Returns
- `Vector`: Actions processed for environment use (e.g., 0-based for Discrete spaces), or raw actions if `raw=true` (if supported)
"""
function predict_actions(agent::ActorCriticAgent, observations::AbstractVector; deterministic::Bool = false, rng::AbstractRNG = agent.rng, raw::Bool = false)
    if raw
        error("ActorCriticAgent does not support raw actions. Use an off-policy actor-critic agent for raw actions.")
    end
    #TODO add !to name?
    layer = agent.layer
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(layer))
    actions, st = predict_actions(layer, batched_obs, ps, st; deterministic = deterministic, rng = rng)
    @reset train_state.states = st
    agent.train_state = train_state
    # Convert policy-space actions to env-space using the agent's adapter
    adapter = agent.action_adapter
    actions = to_env.(Ref(adapter), actions, Ref(action_space(layer)))
    return actions
end

# Abstract methods for all agents
function save_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String = ".jld2")
    error("save_policy_params_and_state not implemented for $(typeof(agent))")
end

function load_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String = ".jld2")
    error("load_policy_params_and_state not implemented for $(typeof(agent))")
end

function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, alg::AbstractAlgorithm)
    return optimizer_type(alg.learning_rate)
end


# Implementation for ActorCriticAgent
function save_policy_params_and_state(agent::ActorCriticAgent, path::AbstractString; suffix::String = ".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Saving policy, parameters, and state to $file_path"
    save(
        file_path, Dict(
            "policy" => agent.policy,
            "parameters" => agent.train_state.parameters,
            "states" => agent.train_state.states
        )
    )
    return file_path
end

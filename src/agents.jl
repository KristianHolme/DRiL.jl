mutable struct AgentStats
    gradient_updates::Int
    steps_taken::Int
end

function add_step!(stats::AgentStats, steps::Int = 1)
    return stats.steps_taken += steps
end

function add_gradient_update!(stats::AgentStats, updates::Int = 1)
    return stats.gradient_updates += updates
end

function steps_taken(stats::AgentStats)
    return stats.steps_taken
end

function gradient_updates(stats::AgentStats)
    return stats.gradient_updates
end

function Random.seed!(agent::AbstractAgent, seed::Integer)
    Random.seed!(agent.rng, seed)
    return agent
end

"""
Agent for Actor-Critic algorithms

    verbose: 
        0: nothing
        1: progress bar
        2: progress bar and stats
        
"""
mutable struct ActorCriticAgent{P <: AbstractActorCriticPolicy, R <: AbstractRNG, L <: AbstractTrainingLogger} <: AbstractAgent
    policy::P
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::L
    verbose::Int
    rng::R
    stats::AgentStats
end

add_step!(agent::ActorCriticAgent, steps::Int = 1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::ActorCriticAgent, updates::Int = 1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::ActorCriticAgent) = steps_taken(agent.stats)
gradient_updates(agent::ActorCriticAgent) = gradient_updates(agent.stats)

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
    policy = agent.policy
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    #FIXME: type instability here, is policy not known??
    actions, values, logprobs, st = policy(batched_obs, ps, st)
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
    policy = agent.policy
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    values, st = predict_values(policy, batched_obs, ps, st)
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

# Returns
- `Vector`: Actions processed for environment use (e.g., 0-based for Discrete spaces)
"""
function predict_actions(agent::ActorCriticAgent, observations::AbstractVector; deterministic::Bool = false, rng::AbstractRNG = agent.rng)
    #TODO add !to name?
    policy = agent.policy
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    actions, st = predict_actions(policy, batched_obs, ps, st; deterministic = deterministic, rng = rng)
    @reset train_state.states = st
    agent.train_state = train_state
    # Process actions for environment use (e.g., convert 1-based to 0-based for Discrete)
    #HACK: incorporate alg into agent?
    alg = PPO()
    actions = process_action.(actions, Ref(action_space(policy)), Ref(alg))
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

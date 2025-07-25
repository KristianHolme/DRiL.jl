mutable struct AgentStats
    gradient_updates::Int
    steps_taken::Int
end

function add_step!(stats::AgentStats, steps::Int=1)
    stats.steps_taken += steps
end

function add_gradient_update!(stats::AgentStats, updates::Int=1)
    stats.gradient_updates += updates
end

function steps_taken(stats::AgentStats)
    return stats.steps_taken
end

function gradient_updates(stats::AgentStats)
    return stats.gradient_updates
end

"""
Agent for Actor-Critic algorithms

    verbose: 
        0: nothing
        1: progress bar
        2: progress bar and stats
        
"""
struct ActorCriticAgent <: AbstractAgent
    policy::AbstractActorCriticPolicy
    train_state::Lux.Training.TrainState
    n_steps::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float32
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing,TensorBoardLogger.TBLogger}
    verbose::Int
    rng::AbstractRNG
    stats::AgentStats
end

add_step!(agent::ActorCriticAgent, steps::Int=1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::ActorCriticAgent, updates::Int=1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::ActorCriticAgent) = steps_taken(agent.stats)
gradient_updates(agent::ActorCriticAgent) = gradient_updates(agent.stats)

function ActorCriticAgent(policy::AbstractActorCriticPolicy;
    n_steps::Int=2048,
    batch_size::Int=64,
    epochs::Int=10,
    learning_rate::Float32=3f-4,
    optimizer_type::Type{<:Optimisers.AbstractRule}=Optimisers.Adam,
    stats_window::Int=100,#TODO not used
    verbose::Int=1,
    log_dir::Union{Nothing,String}=nothing,
    rng::AbstractRNG=Random.default_rng())

    optimizer = make_optimizer(optimizer_type, learning_rate)
    ps, st = Lux.setup(rng, policy)
    # @show ps.log_std
    if !isnothing(log_dir)
        logger = TBLogger(log_dir, tb_increment)
    else
        logger = nothing
    end
    train_state = Lux.Training.TrainState(policy, ps, st, optimizer)
    return ActorCriticAgent(policy, train_state, n_steps, batch_size, epochs,
        learning_rate, optimizer_type, stats_window,
        logger, verbose, rng, AgentStats(0, 0))
end

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
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    actions, values, logprobs, st = policy(batched_obs, ps, st)
    #does this reset work?, probably not
    @reset agent.train_state.states = st
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
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    values, st = predict_values(policy, batched_obs, ps, st)
    #FIXME: this does not work?
    @reset agent.train_state.states = st
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
function predict_actions(agent::ActorCriticAgent, observations::AbstractVector; deterministic::Bool=false, rng::AbstractRNG=agent.rng)
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    # Convert observations vector to batched matrix for policy
    batched_obs = batch(observations, observation_space(policy))
    actions, _ = predict_actions(policy, batched_obs, ps, st; deterministic=deterministic, rng=rng)
    # Process actions for environment use (e.g., convert 1-based to 0-based for Discrete)
    actions = process_action.(actions, Ref(action_space(policy)))
    return actions
end

# Abstract methods for all agents
function save_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String=".jld2")
    error("save_policy_params_and_state not implemented for $(typeof(agent))")
end

function load_policy_params_and_state(agent::AbstractAgent, path::AbstractString; suffix::String=".jld2")
    error("load_policy_params_and_state not implemented for $(typeof(agent))")
end

# Add a helper function for optimizer crtion
function make_optimizer(optimizer_type::Type{<:Optimisers.AbstractRule}, learning_rate::Float32)
    if optimizer_type == Optimisers.Adam
        return optimizer_type(eta=learning_rate, epsilon=1f-5)
    else
        return optimizer_type(learning_rate)
    end
end

# Implementation for ActorCriticAgent
function save_policy_params_and_state(agent::ActorCriticAgent, path::AbstractString; suffix::String=".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Saving policy, parameters, and state to $file_path"
    save(file_path, Dict(
        "policy" => agent.policy,
        "parameters" => agent.train_state.parameters,
        "states" => agent.train_state.states
    ))
    return file_path
end

function load_policy_params_and_state(agent::ActorCriticAgent, path::AbstractString; suffix::String=".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Loading policy, parameters, and state from $file_path"
    data = load(file_path)
    new_policy = data["policy"]
    new_parameters = data["parameters"]
    new_states = data["states"]
    new_optimizer = make_optimizer(agent.optimizer_type, agent.learning_rate)
    new_train_state = Lux.Training.TrainState(new_policy, new_parameters, new_states, new_optimizer)
    #TODO: check if this is correct, probably it is not
    @reset agent.policy = new_policy
    @reset agent.train_state = new_train_state
    return agent
end

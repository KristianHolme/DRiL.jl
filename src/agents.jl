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
struct ActorCriticAgent
    policy::ActorCriticPolicy
    train_state::Lux.Training.TrainState
    n_steps::Int
    batch_size::Int
    epochs::Int
    learning_rate::Float32
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing, TensorBoardLogger.TBLogger}
    verbose::Int
    rng::AbstractRNG
    stats::AgentStats
end

add_step!(agent::ActorCriticAgent, steps::Int=1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::ActorCriticAgent, updates::Int=1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::ActorCriticAgent) = steps_taken(agent.stats)
gradient_updates(agent::ActorCriticAgent) = gradient_updates(agent.stats)

function ActorCriticAgent(policy::ActorCriticPolicy; 
        n_steps::Int=2048, 
        batch_size::Int=64, 
        epochs::Int=10,
        learning_rate::Float32=3f-4,
        optimizer_type::Type{<:Optimisers.AbstractRule}=Optimisers.Adam,
        stats_window::Int=100,#TODO not used
        verbose::Int=1,
        log_dir::Union{Nothing, String}=nothing,
        rng::AbstractRNG=Random.default_rng())


    if optimizer_type == Optimisers.Adam
        optimizer = optimizer_type(eta=learning_rate, epsilon=1f-5)
    else
        optimizer = optimizer_type(learning_rate)
    end
    #TODO add maxgradnorm wrapper
    ps, st = Lux.setup(rng, policy)
    @show ps.log_std
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

function get_action_and_values(agent::ActorCriticAgent, observations::AbstractArray)
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    actions, values, logprobs, st = policy(observations, ps, st)
    @reset agent.train_state.states = st

    return actions, values, logprobs
end

function predict_values(agent::ActorCriticAgent, observations::AbstractArray)
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    critic_st = st.critic_head
    values, critic_st = policy.critic_head(observations, ps.critic_head, critic_st)
    st = merge(st, (;critic_head= critic_st))
    @reset agent.train_state.states = st
    return values
end

function predict_actions(agent::ActorCriticAgent, observations::AbstractArray; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    actions, _ = predict(policy, observations, ps, st; deterministic=deterministic, rng=rng)
    return actions
end

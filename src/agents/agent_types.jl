# Agent type definitions

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
mutable struct ActorCriticAgent{L <: AbstractActorCriticLayer, R <: AbstractRNG, LG <: AbstractTrainingLogger, A <: AbstractAlgorithm, AD <: AbstractActionAdapter} <: AbstractAgent
    layer::L
    algorithm::A
    action_adapter::AD
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::LG
    verbose::Int
    rng::R
    stats::AgentStats
end

add_step!(agent::ActorCriticAgent, steps::Int = 1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::ActorCriticAgent, updates::Int = 1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::ActorCriticAgent) = steps_taken(agent.stats)
gradient_updates(agent::ActorCriticAgent) = gradient_updates(agent.stats)

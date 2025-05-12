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
    stats_window::Int
    verbose::Int
    rng::AbstractRNG
end

function ActorCriticAgent(policy::ActorCriticPolicy; 
        n_steps::Int=2048, 
        batch_size::Int=64, 
        epochs::Int=10,
        learning_rate::Float32=3f-4,
        optimizer::Type{<:Optimisers.AbstractRule}=Optimisers.Adam,
        stats_window::Int=100,
        verbose::Int=1,
        rng::AbstractRNG=Random.default_rng())

    optimizer = optimizer(learning_rate)
    #TODO add maxgradnorm wrapper
    ps, st = Lux.setup(rng, policy)
    train_state = Lux.Training.TrainState(policy, ps, st, optimizer)
    return ActorCriticAgent(policy, train_state, n_steps, batch_size, epochs, stats_window, verbose, rng)
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

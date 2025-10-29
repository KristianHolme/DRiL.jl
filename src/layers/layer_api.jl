# Actor-Critic Layer API (Lux-aligned naming)

"""
    ActorCriticLayer(observation_space, action_space; kwargs...)

Construct an actor-critic layer (training-time) selecting discrete/continuous
variant based on the provided action space.
"""
ActorCriticLayer(observation_space::Union{Discrete, Box}, action_space::Box; kwargs...) =
    ContinuousActorCriticLayer(observation_space, action_space; kwargs...)
ActorCriticLayer(observation_space::Union{Discrete, Box}, action_space::Discrete; kwargs...) =
    DiscreteActorCriticLayer(observation_space, action_space; kwargs...)

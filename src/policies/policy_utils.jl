# Utility functions for policies

noise(policy::ContinuousActorCriticPolicy{<:Any, <:Any, N, <:Any, <:Any, <:Any, <:Any, <:Any}) where {N <: AbstractNoise} = N()
noise(policy::DiscreteActorCriticPolicy{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) = NoNoise()

observation_space(policy::AbstractActorCriticPolicy) = policy.observation_space
action_space(policy::AbstractActorCriticPolicy) = policy.action_space


# Concrete actor-critic layer type definitions

struct ContinuousActorCriticLayer{
        O <: AbstractSpace,
        A <: Box,
        N <: AbstractNoise,
        C <: CriticType,
        F <: FeatureSharing,
        FE <: AbstractLuxLayer,
        AH <: AbstractLuxLayer,
        CH <: AbstractLuxLayer,
        LS,
    } <: AbstractActorCriticLayer
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
    log_std_init::LS
end

struct DiscreteActorCriticLayer{O <: AbstractSpace, A <: Discrete, F <: FeatureSharing, FE <: AbstractLuxLayer, AH <: AbstractLuxLayer, CH <: AbstractLuxLayer} <: AbstractActorCriticLayer
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
end

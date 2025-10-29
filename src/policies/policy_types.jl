# Concrete policy type definitions

struct ContinuousActorCriticPolicy{
        O <: AbstractSpace,
        A <: Box,
        N <: AbstractNoise,
        C <: CriticType,
        F <: FeatureSharing,
        FE <: AbstractLuxLayer,
        AH <: AbstractLuxLayer,
        CH <: AbstractLuxLayer,
        LS,
    } <: AbstractActorCriticPolicy
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
    log_std_init::LS
end

struct DiscreteActorCriticPolicy{O <: AbstractSpace, A <: Discrete, F <: FeatureSharing, FE <: AbstractLuxLayer, AH <: AbstractLuxLayer, CH <: AbstractLuxLayer} <: AbstractActorCriticPolicy
    observation_space::O
    action_space::A
    feature_extractor::FE
    actor_head::AH
    critic_head::CH
end


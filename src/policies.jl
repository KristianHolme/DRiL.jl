abstract type AbstractPolicy <: Lux.AbstractLuxLayer end

#predicts actions from observations, also returns st
function predict(policy::AbstractPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false) end

#returns values, log_probs, entropy, st
function evaluate_actions(policy::AbstractPolicy, obs::AbstractArray, actions::AbstractArray, ps, st) end

#returns actions, values, log_probs, st
function (policy::AbstractPolicy)(obs::AbstractArray, ps, st) end


abstract type AbstractActorCriticPolicy <: AbstractPolicy end

struct ActorCriticPolicy <: AbstractActorCriticPolicy
    observation_space::AbstractSpace
    action_space::AbstractSpace
    feature_extractor::AbstractLuxLayer
    actor_head::AbstractLuxLayer
    critic_head::AbstractLuxLayer
    log_std_init::Union{AbstractFloat, Vector{AbstractFloat}}
end

function ActorCriticPolicy(observation_space::UniformBox, action_space::UniformBox; log_std_init = 0.0f0, hidden_dim = 64, activation = tanh)
    feature_extractor = Lux.FlattenLayer()
    latent_dim = observation_space.shape |> prod
    actor_head = Chain(Dense(latent_dim, hidden_dim, activation), 
                       Dense(hidden_dim, hidden_dim, activation),
                       Dense(hidden_dim, action_space.shape |> prod, activation),
                       ReshapeLayer(action_space.shape))
    critic_head = Chain(Dense(latent_dim, hidden_dim, activation), 
                       Dense(hidden_dim, hidden_dim, activation),
                       Dense(hidden_dim, 1))
    return ActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init)
end

function Lux.initialparameters(rng::AbstractRNG, policy::ActorCriticPolicy)
    params = (feature_extractor = Lux.initialparameters(rng, policy.feature_extractor),
               actor_head = Lux.initialparameters(rng, policy.actor_head),
               critic_head = Lux.initialparameters(rng, policy.critic_head),
               log_std = policy.log_std_init*ones(typeof(policy.log_std_init), policy.action_space.shape))
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::ActorCriticPolicy)
    states = (feature_extractor = Lux.initialstates(rng, policy.feature_extractor),
               actor_head = Lux.initialstates(rng, policy.actor_head),
               critic_head = Lux.initialstates(rng, policy.critic_head))
    return states
end

function Lux.parameterlength(policy::ActorCriticPolicy)
    return Lux.parameterlength(policy.feature_extractor) + Lux.parameterlength(policy.actor_head) + Lux.parameterlength(policy.critic_head) + prod(policy.action_space.shape)
end

function Lux.statelength(policy::ActorCriticPolicy)
    return Lux.statelength(policy.feature_extractor) + Lux.statelength(policy.actor_head) + Lux.statelength(policy.critic_head)
end

function get_distribution_type(policy::ActorCriticPolicy)
    if prod(policy.action_space.shape) > 1
        return Distributions.MvNormal
    else
        return Distributions.Normal
    end
end

function (policy::ActorCriticPolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG=default_rng())
    feats = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    action_mean = policy.actor_head(feats, ps.actor_head, st.actor_head)
    values = policy.critic_head(feats, ps.critic_head, st.critic_head)

    std = exp.(ps.log_std)
    actions, log_probs = get_noisy_actions(policy, action_mean, std, rng; log_probs=true)
    return actions, values, log_probs, st
end

function extract_features(policy::ActorCriticPolicy, obs::AbstractArray, ps, st)
    if ndims(obs) == ndims(policy.observation_space)
        obs = reshape(obs, policy.observation_space.shape..., 1)
    end
    feats, feats_st = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    st = merge(st, (;feature_extractor= feats_st))
    return feats, st
end

function get_action_mean_from_latent(policy::ActorCriticPolicy, latent::AbstractArray, ps, st)
    action_mean, actor_st = policy.actor_head(latent, ps.actor_head, st.actor_head)
    st = merge(st, (;actor_head= actor_st))
    return action_mean, st
end

function get_distributions(distribution_type::Type{<:Distributions.Normal}, action_means::AbstractArray, std::AbstractArray, static_std::Bool)
    if static_std
        return Distributions.Normal.(action_means, Ref(std[1]))
    else
        return Distributions.Normal.(action_means, std)
    end
end

# function get_distributions(distribution_type::Type{<:Distributions.MvNormal}, action_means::AbstractArray, std::AbstractArray, static_std::Bool)
#     if static_std
#         return Distributions.MvNormal.(action_means, Ref(std))
#     else
#         slicedims = ndims(std)
#         return Distributions.MvNormal.(action_means, eachslice(std, dims=slicedims))
#     end
# end

function get_distributions(policy::ActorCriticPolicy, action_means::AbstractArray, std::AbstractArray)
    distribution_type = get_distribution_type(policy)
    static_std = size(std) == size(action_means)
    distributions = get_distributions(distribution_type, action_means, std, static_std)
    return distributions
end

function get_noisy_actions(policy::ActorCriticPolicy, action_mean::AbstractArray, std::AbstractArray, rng::AbstractRNG; log_probs::Bool=false)
    distributions = get_distributions(policy, action_mean, std)
    flattened_actions = rand.(rng, distributions)
    actions = reshape(flattened_actions, policy.action_space.shape..., :)
    if log_probs
        log_probs = loglikelihood.(distributions, flattened_actions)
        return actions, log_probs
    else
        return actions
    end
end

function predict(policy::ActorCriticPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_mean, st = get_action_mean_from_latent(policy, feats, ps, st)
    if deterministic
        return action_mean, st
    else
        std = exp.(ps.log_std)
        actions = get_noisy_actions(policy, action_mean, std, rng; log_probs=false)
        return actions, st
    end
end

function evaluate_actions(policy::ActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_mean, st = get_action_mean_from_latent(policy, feats, ps, st)
    values = policy.critic_head(feats, ps.critic_head, st.critic_head)
    std = exp.(policy.log_std_init)
    distributions = get_distributions(policy, new_action_mean, std)
    log_probs = loglikelihood.(distributions, actions)
    entropy = entropy.(distributions)
    return values, log_probs, entropy, st
end
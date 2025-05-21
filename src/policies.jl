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
    log_std_init::Union{AbstractFloat,Vector{AbstractFloat}}
end

observation_space(policy::ActorCriticPolicy) = policy.observation_space
action_space(policy::ActorCriticPolicy) = policy.action_space

function ActorCriticPolicy(observation_space::UniformBox, action_space::UniformBox; log_std_init=action_space.type(0), hidden_dim=64, activation=tanh)
    feature_extractor = Lux.FlattenLayer()
    latent_dim = observation_space.shape |> prod
    bias_init = zeros32
    hidden_init = (rng, out_dims, in_dims) -> orthogonal(rng, action_space.type, out_dims, in_dims; gain=sqrt(2))
    actor_init = (rng, out_dims, in_dims) -> orthogonal(rng, action_space.type, out_dims, in_dims; gain=0.01)
    value_init = (rng, out_dims, in_dims) -> orthogonal(rng, action_space.type, out_dims, in_dims; gain=1.0)
    actor_head = Chain(Dense(latent_dim, hidden_dim, activation, init_weight=hidden_init, init_bias=bias_init),
        Dense(hidden_dim, hidden_dim, activation, init_weight=hidden_init, init_bias=bias_init),
        Dense(hidden_dim, action_space.shape |> prod, activation, init_weight=actor_init, init_bias=bias_init),
        ReshapeLayer(action_space.shape))
    critic_head = Chain(Dense(latent_dim, hidden_dim, activation, init_weight=hidden_init, init_bias=bias_init),
        Dense(hidden_dim, hidden_dim, activation, init_weight=hidden_init, init_bias=bias_init),
        Dense(hidden_dim, 1, init_weight=value_init, init_bias=bias_init))
    return ActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init)
end

function Lux.initialparameters(rng::AbstractRNG, policy::ActorCriticPolicy)
    params = ComponentArray(feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
        actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head),
        log_std=policy.log_std_init * ones(typeof(policy.log_std_init), policy.action_space.shape))
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::ActorCriticPolicy)
    states = (feature_extractor=Lux.initialstates(rng, policy.feature_extractor),
        actor_head=Lux.initialstates(rng, policy.actor_head),
        critic_head=Lux.initialstates(rng, policy.critic_head))
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

function (policy::ActorCriticPolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_mean, st = get_action_mean_from_latent(policy, feats, ps, st)
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    std = exp.(ps.log_std)
    actions, log_probs = get_noisy_actions(policy, action_mean, std, rng; log_probs=true)
    return actions, values, log_probs, merge(st, (; critic_head=critic_st))
end

function extract_features(policy::ActorCriticPolicy, obs::AbstractArray, ps, st)

    obs_space_shape = observation_space(policy).shape
    obs_shape = size(obs)
    # @info "obs_space_shape: $obs_space_shape"
    # @info "obs_shape: $obs_shape"
    # @info "ndims(obs): $(ndims(obs))"
    # @info "length(obs_space_shape): $(length(obs_space_shape))"
    if ndims(obs) - 1 == length(obs_space_shape)
        batch_size = obs_shape[end]
    else
        batch_size = 1
    end
    reshaped_obs = reshape(obs, obs_space_shape..., batch_size)

    feats, feats_st = policy.feature_extractor(reshaped_obs, ps.feature_extractor, st.feature_extractor)
    st = merge(st, (; feature_extractor=feats_st))
    return feats, st
end

function get_action_mean_from_latent(policy::ActorCriticPolicy, latent::AbstractArray, ps, st)
    action_mean, actor_st = policy.actor_head(latent, ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head=actor_st))
    return action_mean, st
end

function get_distributions(distribution_type::Type{<:Distributions.Normal}, action_means::AbstractArray, std::AbstractArray, static_std::Bool)
    # @show std
    @assert all(std .> 0) "std is not positive"
    if static_std
        return Distributions.Normal.(action_means, Ref(std[1]))
    else
        return Distributions.Normal.(action_means, std)
    end
end

function get_distributions(policy::ActorCriticPolicy, action_means::AbstractArray, std::AbstractArray)
    distribution_type = get_distribution_type(policy)
    static_std = !(size(std) == size(action_means))
    # @info "static_std: $static_std"
    distributions = get_distributions(distribution_type, action_means, std, static_std)
    return distributions
end

# Helper function to process actions: ensure correct type and clipping
function process_action(action::AbstractArray, action_space::UniformBox)
    # First check if type conversion is needed
    if eltype(action) != action_space.type
        @warn "Action type mismatch: $(eltype(action)) != $(action_space.type)"
        action = convert.(action_space.type, action)
    end
    # Then clip to bounds
    action = clamp.(action, action_space.low, action_space.high)
    return action
end

function get_noisy_actions(policy::ActorCriticPolicy, action_means::AbstractArray, std::AbstractArray, rng::AbstractRNG; log_probs::Bool=false)
    # Use reparameterization trick: sample noise from standard normal, then scale and shift
    # This keeps the operation differentiable through the random sampling
    act_shape = size(action_means)
    act_type = action_space(policy).type
    noise = @ignore_derivatives randn(rng, act_type, act_shape...)

    # Apply noise with std: action = mean + std * noise
    actions = action_means + std .* noise

    @assert size(actions) == size(action_means) "action_means and actions have different shapes"


    if log_probs
        # Still need distributions for calculating log probs and entropy
        distributions = get_distributions(policy, action_means, std)
        # Flattened actions for log probability calculation
        flattened_actions = reshape(actions, :, size(actions)[end])
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
        # Process actions: clip and ensure correct type
        actions = action_mean
    else
        std = exp.(ps.log_std)
        @assert all(std .> 0) "std is not positive"
        actions = get_noisy_actions(policy, action_mean, std, rng; log_probs=false)
    end
    #clip/squashing
    actions = process_action(actions, action_space(policy))
    return actions, st
end

function evaluate_actions(policy::ActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_mean, st = get_action_mean_from_latent(policy, feats, ps, st)
    values, st_critic_head = policy.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head=st_critic_head))
    std = exp.(ps.log_std)
    distributions = get_distributions(policy, new_action_mean, std)

    # Make sure actions are correctly shaped for log probability
    flattened_actions = reshape(actions, :, size(actions)[end])

    log_probs = loglikelihood.(distributions, flattened_actions)
    entropy = Distributions.entropy.(distributions)
    return values, log_probs, entropy, st
end
abstract type AbstractPolicy <: Lux.AbstractLuxLayer end

#predicts actions from observations, also returns st
function predict(policy::AbstractPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false) end

#predicts values from observations, returns values and st
function predict_values(policy::AbstractPolicy, obs::AbstractArray, ps, st) end

#returns values, log_probs, entropy, st
function evaluate_actions(policy::AbstractPolicy, obs::AbstractArray, actions::AbstractArray, ps, st) end

#returns actions, values, log_probs, st
function (policy::AbstractPolicy)(obs::AbstractArray, ps, st) end


abstract type AbstractActorCriticPolicy <: AbstractPolicy end

struct ContinuousActorCriticPolicy{O<:AbstractSpace,A<:Box} <: AbstractActorCriticPolicy
    observation_space::O
    action_space::A
    feature_extractor::AbstractLuxLayer
    actor_head::AbstractLuxLayer
    critic_head::AbstractLuxLayer
    log_std_init::Union{AbstractFloat,AbstractArray{AbstractFloat}}
    shared_features::Bool
end

struct DiscreteActorCriticPolicy{O<:AbstractSpace,A<:Discrete} <: AbstractActorCriticPolicy
    observation_space::O
    action_space::A
    feature_extractor::AbstractLuxLayer
    actor_head::AbstractLuxLayer
    critic_head::AbstractLuxLayer
    shared_features::Bool
end


observation_space(policy::AbstractActorCriticPolicy) = policy.observation_space
action_space(policy::AbstractActorCriticPolicy) = policy.action_space

abstract type AbstractWeightInitializer end

struct OrthogonalInitializer{T<:AbstractFloat} <: AbstractWeightInitializer
    gain::T
end

function (init::OrthogonalInitializer{T})(rng::AbstractRNG, out_dims::Int, in_dims::Int) where T
    return orthogonal(rng, T, out_dims, in_dims; gain=init.gain)
end

function get_feature_extractor(O::Box)
    return Lux.FlattenLayer()
end

function get_actor_head(latent_dim::Int, action_dim::Int, hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init)
    layers = []
    if isempty(hidden_dims)
        push!(layers, Dense(latent_dim, action_dim, activation, init_weight=output_init, init_bias=bias_init))
    else
        push!(layers, Dense(latent_dim, hidden_dims[1], activation, init_weight=hidden_init, init_bias=bias_init))
        for i in 2:length(hidden_dims)
            push!(layers, Dense(hidden_dims[i-1], hidden_dims[i], activation, init_weight=hidden_init, init_bias=bias_init))
        end
        push!(layers, Dense(hidden_dims[end], action_dim, init_weight=output_init, init_bias=bias_init))
    end
    return Chain(layers...)
end

function get_actor_head(latent_dim::Int, A::Box, hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init)
    chain = get_actor_head(latent_dim, prod(A.shape), hidden_dims, activation, bias_init, hidden_init, output_init)
    push!(chain, ReshapeLayer(A.shape))
    return chain
end

function get_actor_head(latent_dim::Int, A::Discrete, hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init)
    chain = get_actor_head(latent_dim, A.n, hidden_dims, activation, bias_init, hidden_init, output_init)
    return chain
end

function get_critic_head(laten_dim::Int, hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init)
    layers = []
    if isempty(hidden_dims)
        push!(layers, Dense(laten_dim, 1, init_weight=output_init, init_bias=bias_init))
    else
        push!(layers, Dense(laten_dim, hidden_dims[1], activation, init_weight=hidden_init, init_bias=bias_init))
        for i in 2:length(hidden_dims)
            push!(layers, Dense(hidden_dims[i-1], hidden_dims[i], activation, init_weight=hidden_init, init_bias=bias_init))
        end
        push!(layers, Dense(hidden_dims[end], 1, init_weight=output_init, init_bias=bias_init))
    end
    return Chain(layers...)
end

function ContinuousActorCriticPolicy(observation_space::Box{T}, action_space::Box{T}; log_std_init=T(0), hidden_dim=[64, 64], activation=tanh, shared_features::Bool=true) where T
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = observation_space.shape |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dim, activation, bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, hidden_dim, activation, bias_init, hidden_init, value_init)
    return ContinuousActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init, shared_features)
end

function DiscreteActorCriticPolicy(observation_space::Box{T}, action_space::Discrete; hidden_dim=[64, 64], activation=tanh, shared_features::Bool=true) where T
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = observation_space.shape |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dim, activation, bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, hidden_dim, activation, bias_init, hidden_init, value_init)
    return DiscreteActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, shared_features)
end

# Convenience constructors that maintain the old interface
ActorCriticPolicy(observation_space::Box{T}, action_space::Box{T}; kwargs...) where T = ContinuousActorCriticPolicy(observation_space, action_space; kwargs...)
ActorCriticPolicy(observation_space::Box{T}, action_space::Discrete; kwargs...) where T = DiscreteActorCriticPolicy(observation_space, action_space; kwargs...)

function Lux.initialparameters(rng::AbstractRNG, policy::ContinuousActorCriticPolicy)
    params = ComponentArray(feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
        actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head),
        log_std=policy.log_std_init * ones(typeof(policy.log_std_init), policy.action_space.shape))
    return params
end

function Lux.initialparameters(rng::AbstractRNG, policy::DiscreteActorCriticPolicy)
    params = ComponentArray(feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
        actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head))
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::AbstractActorCriticPolicy)
    states = (feature_extractor=Lux.initialstates(rng, policy.feature_extractor),
        actor_head=Lux.initialstates(rng, policy.actor_head),
        critic_head=Lux.initialstates(rng, policy.critic_head))
    return states
end

function Lux.parameterlength(policy::ContinuousActorCriticPolicy)
    return Lux.parameterlength(policy.feature_extractor) + Lux.parameterlength(policy.actor_head) + Lux.parameterlength(policy.critic_head) + prod(policy.action_space.shape)
end

function Lux.parameterlength(policy::DiscreteActorCriticPolicy)
    return Lux.parameterlength(policy.feature_extractor) + Lux.parameterlength(policy.actor_head) + Lux.parameterlength(policy.critic_head)
end

function Lux.statelength(policy::AbstractActorCriticPolicy)
    return Lux.statelength(policy.feature_extractor) + Lux.statelength(policy.actor_head) + Lux.statelength(policy.critic_head)
end


function (policy::ContinuousActorCriticPolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_mean, st = get_actions_from_latent(policy, feats, ps, st)
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    std = exp.(ps.log_std)
    actions, log_probs = get_noisy_actions(policy, action_mean, std, rng; log_probs=true)
    return actions, values, log_probs, merge(st, (; critic_head=critic_st))
end

function (policy::DiscreteActorCriticPolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_latent(policy, feats, ps, st)  # For discrete, these are logits
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    actions, log_probs = get_discrete_actions(policy, action_logits, rng; log_probs=true, deterministic=false)
    return actions, values, log_probs, merge(st, (; critic_head=critic_st))
end

function extract_features(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)

    obs_space_shape = size(observation_space(policy))
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

function get_actions_from_latent(policy::AbstractActorCriticPolicy, latent::AbstractArray, ps, st)
    actions, actor_st = policy.actor_head(latent, ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head=actor_st))
    return actions, st
end



# For continuous action spaces
function get_distributions(policy::ContinuousActorCriticPolicy, action_means::AbstractArray, std::AbstractArray)
    @assert all(std .> 0) "std is not positive"
    static_std = !(size(std) == size(action_means))

    if prod(policy.action_space.shape) > 1
        # Multivariate case - use MvNormal
        if ndims(action_means) == 1
            # Single observation
            if static_std
                cov = Diagonal(fill(std[1]^2, length(action_means)))
            else
                cov = Diagonal(std .^ 2)
            end
            return Distributions.MvNormal(action_means, cov)
        else
            # Batched observations
            batch_size = size(action_means, ndims(action_means))
            distributions = Vector{Distributions.MvNormal}(undef, batch_size)

            for i in 1:batch_size
                mean_i = action_means[:, i]
                if static_std
                    cov = Diagonal(fill(std[1]^2, length(mean_i)))
                else
                    cov = Diagonal((std[:, i]) .^ 2)
                end
                distributions[i] = Distributions.MvNormal(mean_i, cov)
            end
            return distributions
        end
    else
        # Univariate case - use Normal
        if static_std
            return Distributions.Normal.(action_means, Ref(std[1]))
        else
            return Distributions.Normal.(action_means, std)
        end
    end
end

# For discrete action spaces
function get_distributions(policy::DiscreteActorCriticPolicy, action_logits::AbstractArray)
    # For discrete actions, action_logits are the raw outputs from the network
    # std is not used for discrete actions

    if ndims(action_logits) == 1
        # Single observation
        return Distributions.Categorical(Lux.softmax(action_logits))
    else
        # Batched observations
        batch_size = size(action_logits, ndims(action_logits))
        distributions = Vector{Distributions.Categorical}(undef, batch_size)

        for i in 1:batch_size
            logits_i = selectdim(action_logits, ndims(action_logits), i)
            distributions[i] = Distributions.Categorical(Lux.softmax(logits_i))
        end
        return distributions
    end
end



function get_noisy_actions(policy::AbstractActorCriticPolicy, action_means::AbstractArray, std::AbstractArray, rng::AbstractRNG; log_probs::Bool=false)
    # Use reparameterization trick: sample noise from standard normal, then scale and shift
    # This keeps the operation differentiable through the random sampling
    act_shape = size(action_means)
    act_type = eltype(action_space(policy))
    #TODO is this correct? same noise for all actions?
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

# Action sampling for discrete action spaces
function get_discrete_actions(policy::DiscreteActorCriticPolicy, action_logits::AbstractArray, rng::AbstractRNG; log_probs::Bool=false, deterministic::Bool=false)
    if deterministic
        # For deterministic actions, take the action with highest probability
        if ndims(action_logits) == 1
            # Single observation
            actions = argmax(action_logits) + (policy.action_space.start - 1)  # Convert to action space range
        else
            # Batched observations
            actions = [argmax(selectdim(action_logits, ndims(action_logits), i)) + (policy.action_space.start - 1)
                       for i in 1:size(action_logits, ndims(action_logits))]
        end

        if log_probs
            # Get distributions for log prob calculation
            distributions = get_distributions(policy, action_logits)  # std not used for discrete
            if ndims(action_logits) == 1
                log_prob = logpdf(distributions, actions)
            else
                log_prob = [logpdf(distributions[i], actions[i]) for i in 1:length(actions)]
            end
            return actions, log_prob
        else
            return actions
        end
    else
        # Stochastic sampling
        distributions = get_distributions(policy, action_logits)  # std not used for discrete

        if ndims(action_logits) == 1
            # Single observation
            sampled_action = @ignore_derivatives rand(rng, distributions)
            actions = sampled_action + (policy.action_space.start - 1)  # Convert to action space range
        else
            # Batched observations
            sampled_actions = [@ignore_derivatives rand(rng, distributions[i]) for i in 1:length(distributions)]
            actions = [a + (policy.action_space.start - 1) for a in sampled_actions]  # Convert to action space range
        end

        if log_probs
            if ndims(action_logits) == 1
                log_prob = logpdf(distributions, sampled_action)
            else
                log_prob = [logpdf(distributions[i], sampled_actions[i]) for i in 1:length(sampled_actions)]
            end
            return actions, log_prob
        else
            return actions
        end
    end
end

function predict(policy::ContinuousActorCriticPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
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

function predict(policy::DiscreteActorCriticPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_action_mean_from_latent(policy, feats, ps, st)  # For discrete, these are logits

    actions = get_discrete_actions(policy, action_logits, rng; log_probs=false, deterministic=deterministic)
    # Process actions: ensure they're in the correct range
    actions = process_action(actions, action_space(policy))
    return actions, st
end

function evaluate_actions(policy::ContinuousActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
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

function evaluate_actions(policy::DiscreteActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_logits, st = get_action_mean_from_latent(policy, feats, ps, st)  # For discrete, these are logits
    values, st_critic_head = policy.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head=st_critic_head))

    distributions = get_distributions(policy, new_action_logits, Float32[])  # std not used for discrete

    # Convert actions back to 1-based indexing for distribution evaluation
    # actions are in action_space range, need to convert to distribution range (1-based)
    dist_actions = actions .- (policy.action_space.start - 1)

    if ndims(new_action_logits) == 1
        # Single observation
        log_probs = [logpdf(distributions, dist_actions)]
        entropy = [Distributions.entropy(distributions)]
    else
        # Batched observations  
        log_probs = [logpdf(distributions[i], dist_actions[i]) for i in 1:length(distributions)]
        entropy = [Distributions.entropy(distributions[i]) for i in 1:length(distributions)]
    end

    return values, log_probs, entropy, st
end

function predict_values(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head=critic_st))
    return values, st
end
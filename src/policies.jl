abstract type AbstractPolicy <: Lux.AbstractLuxLayer end

"""
    predict_actions(policy::AbstractPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false) -> (actions, st)

Predict actions from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `ps`: Policy parameters
- `st`: Policy state
- `deterministic::Bool=false`: Whether to use deterministic actions

# Returns
- `actions`: Vector/Array of actions (raw policy outputs, not processed for environment)
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
- Use `process_action()` to convert for environment use
"""
function predict_actions end

"""
    predict_values(policy::AbstractPolicy, obs::AbstractArray, ps, st) -> (values, st)

Predict value estimates from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `values`: Vector of value estimates
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
"""
function predict_values end

"""
    evaluate_actions(policy::AbstractPolicy, obs::AbstractArray, actions::AbstractArray, ps, st) -> (values, log_probs, entropy, st)

Evaluate given actions for batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `actions::AbstractArray`: Batched actions to evaluate (raw policy format)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities for the actions
- `entropy`: Vector of policy entropy values
- `st`: Updated policy state

# Notes
- All inputs must be batched (matrix/array format)
- Actions should be in raw policy format (e.g., 1-based for Discrete)
"""
function evaluate_actions end

"""
    (policy::AbstractPolicy)(obs::AbstractArray, ps, st) -> (actions, values, log_probs, st)

Forward pass through policy: get actions, values, and log probabilities from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (each column is one observation)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `actions`: Vector/Array of actions (raw policy outputs)
- `values`: Vector of value estimates
- `log_probs`: Vector of log probabilities
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
"""
function (policy::AbstractPolicy) end

abstract type AbstractNoise end

struct StateIndependantNoise <: AbstractNoise end
struct NoNoise <: AbstractNoise end

abstract type AbstractActorCriticPolicy <: AbstractPolicy end

struct ContinuousActorCriticPolicy{O<:AbstractSpace,A<:Box,N<:AbstractNoise} <: AbstractActorCriticPolicy
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

noise(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N}) where N<:AbstractNoise = N()
noise(policy::DiscreteActorCriticPolicy{<:Any,<:Any}) = NoNoise()


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

function get_feature_extractor(O::Discrete)
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
    chain = Chain(chain, ReshapeLayer(size(A)))
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

function ContinuousActorCriticPolicy(observation_space::Union{Discrete,Box{T}}, action_space::Box{T}; log_std_init=T(0), hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true) where T
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation, bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, hidden_dims, activation, bias_init, hidden_init, value_init)
    return ContinuousActorCriticPolicy{typeof(observation_space),typeof(action_space),StateIndependantNoise}(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init, shared_features)
end

function DiscreteActorCriticPolicy(observation_space::Union{Discrete,Box}, action_space::Discrete; hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true)
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{Float32}(sqrt(Float32(2)))
    actor_init = OrthogonalInitializer{Float32}(Float32(0.01))
    value_init = OrthogonalInitializer{Float32}(Float32(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation, bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, hidden_dims, activation, bias_init, hidden_init, value_init)
    return DiscreteActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, shared_features)
end


# Convenience constructors that maintain the old interface
ActorCriticPolicy(observation_space::Union{Discrete,Box}, action_space::Box; kwargs...) = ContinuousActorCriticPolicy(observation_space, action_space; kwargs...)
ActorCriticPolicy(observation_space::Union{Discrete,Box}, action_space::Discrete; kwargs...) = DiscreteActorCriticPolicy(observation_space, action_space; kwargs...)

function Lux.initialparameters(rng::AbstractRNG, policy::ContinuousActorCriticPolicy)
    params = ComponentArray(feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
        actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head),
        log_std=policy.log_std_init * ones(typeof(policy.log_std_init), size(policy.action_space)))
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


function (policy::ContinuousActorCriticPolicy)(obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, feats, ps, st)
    values, st = get_values_from_features(policy, feats, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

function (policy::DiscreteActorCriticPolicy)(obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, feats, ps, st)
    ds = get_distributions(policy, action_logits)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

function extract_features(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    feats, feats_st = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    st = merge(st, (; feature_extractor=feats_st))
    return feats, st
end

function get_actions_from_features(policy::AbstractActorCriticPolicy, feats::AbstractArray, ps, st)
    actions, actor_st = policy.actor_head(feats, ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head=actor_st))
    return actions, st
end

function get_values_from_features(policy::AbstractActorCriticPolicy, feats::AbstractArray, ps, st)
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head=critic_st))
    return values, st
end

# For continuous action spaces
#TODO: dispatch in noise type?
function get_distributions(policy::ContinuousActorCriticPolicy, action_means::AbstractArray, log_std::AbstractArray)
    # static_std = !(size(std) == size(action_means))
    batch_dim = ndims(action_means)
    noise_type = noise(policy)
    if noise_type == StateIndependantNoise()
        return DiagGaussian.(eachslice(action_means, dims=batch_dim), Ref(log_std))
    else
        @assert size(log_std) == size(action_means) "log_std and action_means have different shapes"
        return DiagGaussian.(eachslice(action_means, dims=batch_dim), eachslice(log_std, dims=batch_dim))
    end
end

# For discrete action spaces
function get_distributions(policy::DiscreteActorCriticPolicy, action_logits::AbstractArray)
    # For discrete actions, action_logits are the raw outputs from the network
    # std is not used for discrete actions
    probs = Lux.softmax(action_logits)
    batch_dim = ndims(action_logits)
    start = action_space(policy).start
    return Categorical.(eachslice(probs, dims=batch_dim), start)
end

function predict_actions(policy::ContinuousActorCriticPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, feats, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    if deterministic
        actions = mode.(ds)
    else
        actions = rand.(rng, ds)
    end
    return actions, st
end

function predict_actions(policy::DiscreteActorCriticPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
    feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, feats, ps, st)  # For discrete, these are logits
    ds = get_distributions(policy, action_logits)
    if deterministic
        actions = mode.(ds)
    else
        actions = rand.(rng, ds)
    end
    return actions, st
end

function evaluate_actions(policy::ContinuousActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_means, st = get_actions_from_features(policy, feats, ps, st)
    values, st = get_values_from_features(policy, feats, ps, st)
    distributions = get_distributions(policy, new_action_means, ps.log_std)
    log_probs = logpdf.(distributions, eachslice(actions, dims=ndims(actions)))
    entropies = entropy.(distributions)
    return vec(values), log_probs, entropies, st
end

function evaluate_actions(policy::DiscreteActorCriticPolicy, obs::AbstractArray, actions::AbstractArray{<:Int}, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_logits, st = get_actions_from_features(policy, feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, feats, ps, st)
    ds = get_distributions(policy, new_action_logits)
    log_probs = logpdf.(ds, eachslice(actions, dims=ndims(actions)))
    entropies = entropy.(ds)
    return vec(values), log_probs, entropies, st
end

function predict_values(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, feats, ps, st)
    return vec(values), st
end
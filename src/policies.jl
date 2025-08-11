abstract type AbstractPolicy <: Lux.AbstractLuxLayer end
abstract type CriticType end
@kwdef struct QCritic <: CriticType
    n_critics::Int = 2
end
struct VCritic <: CriticType end

"""
    predict_actions(policy::AbstractPolicy, obs::AbstractArray, ps, st; deterministic::Bool=false) -> (actions, st)

Predict actions from batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
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
    predict_values(policy::AbstractPolicy, obs::AbstractArray, [actions::AbstractArray,] ps, st) -> (values, st)

Predict Q-values from batched observations and actions (for Q-Critic policies).

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `actions::AbstractArray`: Batched actions (last dimension is batch) (only for Q-Critic policies)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `values`: batched values (tuples of values for multiple Q-Critic networks)
- `st`: Updated policy state

# Notes
- Input observations and actions must be batched (matrix/array format)
- Actions should be in raw policy format (e.g., 1-based for Discrete)
"""
function predict_values end


"""
    evaluate_actions(policy::AbstractPolicy, obs::AbstractArray, actions::AbstractArray, ps, st) -> (values, log_probs, entropy, st)

Evaluate given actions for batched observations.

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
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
    action_log_prob(policy::AbstractPolicy, obs::AbstractArray, ps, st) -> (actions, log_probs, st)

Sample actions and return their log probabilities from batched observations (for SAC).

# Arguments
- `policy::AbstractPolicy`: The policy
- `obs::AbstractArray`: Batched observations (last dimension is batch)
- `ps`: Policy parameters
- `st`: Policy state

# Returns
- `actions`: Vector/Array of sampled actions
- `log_probs`: Vector of log probabilities for the sampled actions
- `st`: Updated policy state

# Notes
- Input observations must be batched (matrix/array format)
- Output actions are raw policy outputs (e.g., 1-based for Discrete policies)
"""
function action_log_prob end

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
struct StateDependentNoise <: AbstractNoise end
struct NoNoise <: AbstractNoise end

abstract type AbstractActorCriticPolicy <: AbstractPolicy end

#TODO: add critic_type to parameters?
struct ContinuousActorCriticPolicy{
    O<:AbstractSpace,
    A<:Box,
    N<:AbstractNoise,
    C<:CriticType} <: AbstractActorCriticPolicy
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

noise(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,<:Any}) where N<:AbstractNoise = N()
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

function get_mlp(latent_dim::Int, output_dim::Int, hidden_dims::Vector{Int}, activation::Function,
    bias_init, hidden_init, output_init
)
    layers = []
    if isempty(hidden_dims)
        push!(layers, Dense(latent_dim, 1, init_weight=output_init, init_bias=bias_init))
    else
        push!(layers, Dense(latent_dim, hidden_dims[1], activation, init_weight=hidden_init,
            init_bias=bias_init))
        for i in 2:length(hidden_dims)
            push!(layers, Dense(hidden_dims[i-1], hidden_dims[i], activation,
                init_weight=hidden_init, init_bias=bias_init))
        end
        push!(layers, Dense(hidden_dims[end], output_dim, init_weight=output_init,
            init_bias=bias_init))
    end
    return Chain(layers...)
end

function get_actor_head(latent_dim::Int, action_dim::Int, hidden_dims::Vector{Int},
    activation::Function, bias_init, hidden_init, output_init
)
    return get_mlp(latent_dim, action_dim, hidden_dims, activation, bias_init, hidden_init,
        output_init)
end

function get_actor_head(latent_dim::Int, A::Box, hidden_dims::Vector{Int},
    activation::Function, bias_init, hidden_init, output_init
)
    chain = get_actor_head(latent_dim, prod(size(A)), hidden_dims, activation, bias_init,
        hidden_init, output_init)
    chain = Chain(chain, ReshapeLayer(size(A)))
    return chain
end

function get_actor_head(latent_dim::Int, A::Discrete, hidden_dims::Vector{Int},
    activation::Function, bias_init, hidden_init, output_init
)
    chain = get_actor_head(latent_dim, A.n, hidden_dims, activation, bias_init,
        hidden_init, output_init)
    return chain
end


function get_critic_head(latent_dim::Int, action_space::Box, hidden_dims::Vector{Int},
    activation::Function, bias_init, hidden_init, output_init, critic_type::QCritic
)
    action_dim = size(action_space) |> prod
    mlp = get_mlp(latent_dim + action_dim, 1, hidden_dims, activation, bias_init, hidden_init,
        output_init)
    net = Lux.Parallel(vcat, [mlp for _ in 1:critic_type.n_critics]...)
    return net
end

function get_critic_head(latent_dim::Int, action_space::AbstractSpace,
    hidden_dims::Vector{Int}, activation::Function, bias_init, hidden_init, output_init,
    critic_type::VCritic
)
    return get_mlp(latent_dim, 1, hidden_dims, activation, bias_init, hidden_init, output_init)
end

function ContinuousActorCriticPolicy(observation_space::Union{Discrete,Box{T}},
    action_space::Box{T};
    log_std_init=T(0),
    hidden_dims=[64, 64],
    activation=tanh,
    shared_features::Bool=true,
    critic_type::CriticType=VCritic()
) where T

    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, value_init, critic_type)
    return ContinuousActorCriticPolicy{
        typeof(observation_space),
        typeof(action_space),
        StateIndependantNoise,
        typeof(critic_type)
    }(
        observation_space,
        action_space,
        feature_extractor,
        actor_head,
        critic_head,
        log_std_init,
        shared_features
    )
end

function DiscreteActorCriticPolicy(observation_space::Union{Discrete,Box},
    action_space::Discrete; hidden_dims=[64, 64], activation=tanh,
    shared_features::Bool=true
)
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{Float32}(sqrt(Float32(2)))
    actor_init = OrthogonalInitializer{Float32}(Float32(0.01))
    value_init = OrthogonalInitializer{Float32}(Float32(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, action_space, hidden_dims, activation,
        bias_init, hidden_init, value_init, QCritic())
    return DiscreteActorCriticPolicy(observation_space, action_space, feature_extractor,
        actor_head, critic_head, shared_features)
end


# Convenience constructors that maintain the old interface
ActorCriticPolicy(observation_space::Union{Discrete,Box}, action_space::Box; kwargs...) =
    ContinuousActorCriticPolicy(observation_space, action_space; kwargs...)
ActorCriticPolicy(observation_space::Union{Discrete,Box}, action_space::Discrete; kwargs...) =
    DiscreteActorCriticPolicy(observation_space, action_space; kwargs...)

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.initialparameters(rng::AbstractRNG, policy::ContinuousActorCriticPolicy)
    if policy.shared_features
        feats_params = (feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),)
    else
        feats_params = (actor_feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
            critic_feature_extractor=Lux.initialparameters(rng, policy.feature_extractor))
    end
    head_params = (actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head),
        log_std=policy.log_std_init *
                ones(typeof(policy.log_std_init), size(policy.action_space)))
    params = merge(feats_params, head_params)
    params = ComponentArray(params)
    return params
end

function Lux.initialparameters(rng::AbstractRNG, policy::DiscreteActorCriticPolicy)
    if policy.shared_features
        feats_params = (feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),)
    else
        feats_params = (actor_feature_extractor=Lux.initialparameters(rng, policy.feature_extractor),
            critic_feature_extractor=Lux.initialparameters(rng, policy.feature_extractor))
    end
    head_params = (actor_head=Lux.initialparameters(rng, policy.actor_head),
        critic_head=Lux.initialparameters(rng, policy.critic_head))
    params = merge(feats_params, head_params)
    params = ComponentArray(params)
    return params
end

function Lux.initialstates(rng::AbstractRNG, policy::AbstractActorCriticPolicy)
    if policy.shared_features
        feats_states = (feature_extractor=Lux.initialstates(rng, policy.feature_extractor),)
    else
        feats_states = (actor_feature_extractor=Lux.initialstates(rng, policy.feature_extractor),
            critic_feature_extractor=Lux.initialstates(rng, policy.feature_extractor))
    end
    head_states = (actor_head=Lux.initialstates(rng, policy.actor_head),
        critic_head=Lux.initialstates(rng, policy.critic_head))
    states = merge(feats_states, head_states)
    return states
end

#TODO: add ent_coef as parameter for Q-value critics?
function Lux.parameterlength(policy::ContinuousActorCriticPolicy)
    if policy.shared_features
        feats_len = Lux.parameterlength(policy.feature_extractor)
    else
        feats_len = Lux.parameterlength(policy.feature_extractor) +
                    Lux.parameterlength(policy.feature_extractor)
    end
    head_len = Lux.parameterlength(policy.actor_head) +
               Lux.parameterlength(policy.critic_head)
    total_len = feats_len + head_len + prod(policy.action_space.shape)
    return total_len
end

function Lux.parameterlength(policy::DiscreteActorCriticPolicy)
    if policy.shared_features
        feats_len = Lux.parameterlength(policy.feature_extractor)
    else
        feats_len = Lux.parameterlength(policy.feature_extractor) +
                    Lux.parameterlength(policy.feature_extractor)
    end
    head_len = Lux.parameterlength(policy.actor_head) +
               Lux.parameterlength(policy.critic_head)
    return feats_len + head_len
end

function Lux.statelength(policy::AbstractActorCriticPolicy)
    if policy.shared_features
        feats_len = Lux.statelength(policy.feature_extractor)
    else
        feats_len = Lux.statelength(policy.feature_extractor) +
                    Lux.statelength(policy.feature_extractor)
    end
    head_len = Lux.statelength(policy.actor_head) +
               Lux.statelength(policy.critic_head)
    return feats_len + head_len
end


function (policy::ContinuousActorCriticPolicy)(obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

function (policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,QCritic})(obs::AbstractArray,
    actions::AbstractArray, ps, st
) where N<:AbstractNoise
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
    values, st = get_values_from_features(policy, critic_feats, actions, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, values, log_probs, st
end

function (policy::DiscreteActorCriticPolicy)(obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, actor_feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    ds = get_distributions(policy, action_logits)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

function extract_features(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    if policy.shared_features
        feats, feats_st = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
        actor_feats = feats
        critic_feats = feats
        st = merge(st, (; feature_extractor=feats_st))
    else
        actor_feats, actor_feats_st = policy.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
        critic_feats, critic_feats_st = policy.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
        st = merge(st, (; actor_feature_extractor=actor_feats_st, critic_feature_extractor=critic_feats_st))
    end
    return actor_feats, critic_feats, st
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

function get_values_from_features(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,QCritic}, feats::AbstractArray, actions::AbstractArray, ps, st) where N<:AbstractNoise
    if ndims(actions) == 1
        actions = batch(actions, action_space(policy))
    end
    inputs = vcat(feats, actions)
    values, critic_st = policy.critic_head(inputs, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head=critic_st))
    return values, st
end

# For continuous action spaces
# Dispatch on noise type for VCritic policies
function get_distributions(policy::ContinuousActorCriticPolicy{<:Any,<:Any,StateIndependantNoise,VCritic}, action_means::AbstractArray, log_std::AbstractArray)
    batch_dim = ndims(action_means)
    return DiagGaussian.(eachslice(action_means, dims=batch_dim), Ref(log_std))
end

function get_distributions(policy::ContinuousActorCriticPolicy{<:Any,<:Any,StateDependentNoise,VCritic}, action_means::AbstractArray, log_std::AbstractArray)
    batch_dim = ndims(action_means)
    @assert size(log_std) == size(action_means) "log_std and action_means have different shapes"
    return DiagGaussian.(eachslice(action_means, dims=batch_dim), eachslice(log_std, dims=batch_dim))
end

# Dispatch on noise type for QCritic policies
function get_distributions(policy::ContinuousActorCriticPolicy{<:Any,<:Any,StateIndependantNoise,QCritic}, action_means::AbstractArray, log_std::AbstractArray)
    batch_dim = ndims(action_means)
    return SquashedDiagGaussian.(eachslice(action_means, dims=batch_dim), Ref(log_std))
end

function get_distributions(policy::ContinuousActorCriticPolicy{<:Any,<:Any,StateDependentNoise,QCritic}, action_means::AbstractArray, log_std::AbstractArray)
    batch_dim = ndims(action_means)
    @assert size(log_std) == size(action_means) "log_std and action_means have different shapes"
    return SquashedDiagGaussian.(eachslice(action_means, dims=batch_dim), eachslice(log_std, dims=batch_dim))
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
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
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
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, actor_feats, ps, st)  # For discrete, these are logits
    ds = get_distributions(policy, action_logits)
    if deterministic
        actions = mode.(ds)
    else
        actions = rand.(rng, ds)
    end
    return actions, st
end

function evaluate_actions(policy::ContinuousActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    new_action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    distributions = get_distributions(policy, new_action_means, ps.log_std)
    log_probs = logpdf.(distributions, eachslice(actions, dims=ndims(actions)))
    entropies = entropy.(distributions)
    return evaluate_actions_returns(policy, values, log_probs, entropies, st)
end

function evaluate_actions_returns(::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, values, log_probs, entropies, st)
    return values, log_probs, entropies, st #dont return vec(values) as values is a matrix
end
function evaluate_actions_returns(::ContinuousActorCriticPolicy, values, log_probs, entropies, st)
    return vec(values), log_probs, entropies, st
end

function evaluate_actions(policy::DiscreteActorCriticPolicy, obs::AbstractArray, actions::AbstractArray{<:Int}, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    new_action_logits, st = get_actions_from_features(policy, actor_feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    ds = get_distributions(policy, new_action_logits)
    log_probs = logpdf.(ds, eachslice(actions, dims=ndims(actions)))
    entropies = entropy.(ds)
    return vec(values), log_probs, entropies, st
end

function predict_values(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    return vec(values), st
end

function predict_values(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,QCritic}, obs::AbstractArray, actions::AbstractArray, ps, st) where N<:AbstractNoise
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, critic_feats, actions, ps, st)
    return values, st #dont return vec(values) as this is a matrix
end

#returns vector of actions
function action_log_prob(policy::ContinuousActorCriticPolicy, obs::AbstractArray, ps, st; rng::AbstractRNG=Random.default_rng())
    actor_feats, _, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    actions = rand.(rng, ds)
    log_probs = logpdf.(ds, actions)
    # scaled_actions = scale_to_space.(actions, Ref(policy.action_space))
    return actions, log_probs, st
end

function zero_critic_grads!(critic_grad::ComponentArray, policy::ContinuousActorCriticPolicy)
    if policy.shared_features
        names_to_zero = [:critic_head]
    else
        names_to_zero = [:critic_head, :critic_feature_extractor]
    end
    zero_fields!(critic_grad, names_to_zero)
    nothing
end

function zero_fields!(a::ComponentArray{T}, names::Vector{Symbol}) where T<:Real
    for name in names
        a[name] .= zero(T)
    end
    nothing
end

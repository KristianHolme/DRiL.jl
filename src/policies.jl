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

struct ContinuousActorCriticPolicy{O<:AbstractSpace,A<:Box, N<:AbstractNoise} <: AbstractActorCriticPolicy
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

noise(policy::ContinuousActorCriticPolicy{<:Any, <:Any, N}) where N <: AbstractNoise = N()
noise(policy::DiscreteActorCriticPolicy{<:Any, <:Any})  = NoNoise()


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

function ContinuousActorCriticPolicy(observation_space::Union{Discrete, Box{T}}, action_space::Box{T}; log_std_init=T(0), hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true) where T
    feature_extractor = get_feature_extractor(observation_space)
    latent_dim = size(observation_space) |> prod
    #TODO: make this bias init work for different types
    bias_init = zeros32

    hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
    actor_init = OrthogonalInitializer{T}(T(0.01))
    value_init = OrthogonalInitializer{T}(T(1.0))
    actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation, bias_init, hidden_init, actor_init)
    critic_head = get_critic_head(latent_dim, hidden_dims, activation, bias_init, hidden_init, value_init)
    return ContinuousActorCriticPolicy{typeof(observation_space), typeof(action_space), StateIndependantNoise}(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init, shared_features)
end
#TODO remove
# function ContinuousActorCriticPolicy(observation_space::Discrete, action_space::Box{T}; log_std_init=T(0), hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true) where T
#     feature_extractor = get_feature_extractor(observation_space)
#     latent_dim = size(observation_space) |> prod
#     #TODO: make this bias init work for different types
#     bias_init = zeros32

#     hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
#     actor_init = OrthogonalInitializer{T}(T(0.01))
#     value_init = OrthogonalInitializer{T}(T(1.0))
#     actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation, bias_init, hidden_init, actor_init)
#     critic_head = get_critic_head(latent_dim, hidden_dims, activation, bias_init, hidden_init, value_init)
#     return ContinuousActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, log_std_init, shared_features, StateIndependantNoise())
# end

function DiscreteActorCriticPolicy(observation_space::Union{Discrete, Box}, action_space::Discrete; hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true)
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

#TODO remove
# function DiscreteActorCriticPolicy(observation_space::Discrete, action_space::Discrete; hidden_dims=[64, 64], activation=tanh, shared_features::Bool=true)
#     feature_extractor = get_feature_extractor(observation_space)
#     latent_dim = size(observation_space) |> prod
#     #TODO: make this bias init work for different types
#     bias_init = zeros32

#     # For Discrete observation spaces, we'll use Float32 as the default type
#     T = Float32
#     hidden_init = OrthogonalInitializer{T}(sqrt(T(2)))
#     actor_init = OrthogonalInitializer{T}(T(0.01))
#     value_init = OrthogonalInitializer{T}(T(1.0))
#     actor_head = get_actor_head(latent_dim, action_space, hidden_dims, activation, bias_init, hidden_init, actor_init)
#     critic_head = get_critic_head(latent_dim, hidden_dims, activation, bias_init, hidden_init, value_init)
#     return DiscreteActorCriticPolicy(observation_space, action_space, feature_extractor, actor_head, critic_head, shared_features, StateIndependantNoise())
# end

# Convenience constructors that maintain the old interface
ActorCriticPolicy(observation_space::Union{Discrete, Box}, action_space::Box; kwargs...) = ContinuousActorCriticPolicy(observation_space, action_space; kwargs...)
ActorCriticPolicy(observation_space::Union{Discrete, Box}, action_space::Discrete; kwargs...) = DiscreteActorCriticPolicy(observation_space, action_space; kwargs...)
# ActorCriticPolicy(observation_space::Discrete, action_space::Box{T}; kwargs...) where T = ContinuousActorCriticPolicy(observation_space, action_space; kwargs...)
# ActorCriticPolicy(observation_space::Discrete, action_space::Discrete; kwargs...) = DiscreteActorCriticPolicy(observation_space, action_space; kwargs...)

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
    actions = mode.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

function (policy::DiscreteActorCriticPolicy)(obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, feats, ps, st)
    ds = get_distributions(policy, action_logits)
    actions = mode.(ds)
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

    # if prod(policy.action_space.shape) > 1
    #     # Multivariate case - use MvNormal
        
    #     # Batched observations
    #     batch_size = size(action_means, ndims(action_means))
    #     distributions = Vector{Distributions.MvNormal}(undef, batch_size)

    #     for i in 1:batch_size
    #         mean_i = action_means[:, i]
    #         if static_std
    #             cov = Diagonal(fill(std[1]^2, length(mean_i)))
    #         else
    #             cov = Diagonal((std[:, i]) .^ 2)
    #         end
    #         distributions[i] = Distributions.MvNormal(mean_i, cov)
    #     end
    #     return distributions
    # else
    #     # Univariate case - use Normal
    #     if static_std
    #         return Distributions.Normal.(action_means, Ref(std[1]))
    #     else
    #         return Distributions.Normal.(action_means, std)
    #     end
    # end
end

# For discrete action spaces
function get_distributions(::DiscreteActorCriticPolicy, action_logits::AbstractArray)
    # For discrete actions, action_logits are the raw outputs from the network
    # std is not used for discrete actions
    probs = Lux.softmax(action_logits)
    batch_dim = ndims(action_logits)
    return Categorical.(eachslice(probs, dims=batch_dim))

    # if ndims(action_logits) == 1
    #     # Single observation
    #     return Distributions.Categorical(Lux.softmax(action_logits))
    # else
    #     # Batched observations
    #     batch_size = size(action_logits, ndims(action_logits))
    #     probs = Lux.softmax(action_logits)
    #     #TODO simplify this when PR is done https://github.com/JuliaStats/Distributions.jl/pull/1908
    #     # distributions = Distributions.Categorical.(eachcol(probs))
    #     vec_probs = eachcol(probs) .|> Vector
    #     distributions = Distributions.Categorical.(vec_probs)
    #     return distributions
    # end
end



# function get_noisy_actions(policy::AbstractActorCriticPolicy, action_means::AbstractArray, std::AbstractArray, rng::AbstractRNG; log_probs::Bool=false)
#     # Use reparameterization trick: sample noise from standard normal, then scale and shift
#     # This keeps the operation differentiable through the random sampling
#     act_shape = size(action_means)
#     act_type = eltype(action_space(policy))
#     #TODO is this correct? same noise for all actions?
#     noise = @ignore_derivatives randn(rng, act_type, act_shape...)

#     # Apply noise with std: action = mean + std * noise
#     actions = action_means .+ std .* noise

#     @assert size(actions) == size(action_means) "action_means and actions have different shapes"


#     if log_probs
#         # Still need distributions for calculating log probs and entropy
#         distributions = get_distributions(policy, action_means, std)
#         # Flattened actions for log probability calculation
#         flattened_actions = reshape(actions, :, size(actions)[end])
#         log_probs = loglikelihood.(distributions, flattened_actions)
#         return actions, log_probs
#     else
#         return actions
#     end
# end

# Action sampling for discrete action spaces
# function get_discrete_actions(policy::DiscreteActorCriticPolicy, action_logits::AbstractArray, rng::AbstractRNG; log_probs::Bool=false, deterministic::Bool=false)
#     if deterministic
#         # For deterministic actions, take the action with highest probability
#         # Batched observations - keep in 1-based indexing
#         actions = argmax.(eachcol(action_logits))
#     else
#         # Stochastic sampling
#         distributions = @ignore_derivatives get_distributions(policy, action_logits)
#         # Batched observations - sampled actions are naturally 1-based
#         actions = @ignore_derivatives rand.(rng, distributions)
#     end
#     if log_probs
#         # Use broadcasting for log prob calculation
#         log_probs_matrix = Lux.logsoftmax(action_logits)

#         action_indices = Int.(actions)
#         log_prob = getindex.(eachcol(log_probs_matrix), action_indices)

#         actions = reshape(actions, (1, size(action_logits, 2)))
#         log_prob = reshape(log_prob, (1, size(action_logits, 2)))
#         return actions, log_prob
#     else
#         actions = reshape(actions, (1, size(action_logits, 2)))
#         return actions
#     end
# end

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

function calculate_log_probs(action_mean, log_std, action)
    # Calculate log probability for diagonal Gaussian distribution
    # action_mean: mean vector
    # log_std: log standard deviation vector 
    # action: action vector
    
    # Compute difference from mean
    diff = action .- action_mean
    mock_actions = similar(action).*0 .+ 1
    action2 = @ignore_derivatives mock_actions
    diff = mock_actions .- action_mean
    diff = action2 .- action_mean
    
    # Calculate log probability components
    log_2pi = Float32(log(2π))
    variance_term = sum(2 .* log_std)
    quadratic_term = sum((diff .* diff) )
    
    # Sum components for final log probability
    log_prob = -0.5f0 * (log_2pi + variance_term + quadratic_term)
    

    log_prob = mean(diff.^2)
    # Sum across action dimensions
    @assert log_prob isa Float32
    return log_prob
end

function calculate_entropy(log_std)
    # Calculate entropy for diagonal Gaussian distribution
    # log_std: log standard deviation vector
    log_2pi = Float32(log(2π))
    return sum(0.5f0 .* (log_2pi .+ 2 .* log_std))
end

function evaluate_actions(policy::ContinuousActorCriticPolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    new_action_means, st = get_actions_from_features(policy, feats, ps, st)
    # @info "new_action_means: $(new_action_means)"
    # @info "actions: $(actions)"
    values, st = get_values_from_features(policy, feats, ps, st)

    no_grad_actions = @ignore_derivatives actions

    distributions = get_distributions(policy, new_action_means, ps.log_std)
    log_probs = logpdf.(distributions, eachslice(no_grad_actions, dims=ndims(no_grad_actions)))
    entropies = entropy.(distributions)

    # log_probs = calculate_log_probs.(eachslice(new_action_means, dims=ndims(new_action_means)), Ref(ps.log_std), eachslice(actions, dims=ndims(actions)))
    # @assert length(log_probs) == size(actions, ndims(actions))
    # ent = calculate_entropy(ps.log_std)
    # entropies = [copy(ent) for _ in 1:size(actions, ndims(actions))]

    # with this (WRONG), the gradient is not zero
    # log_probs = vec(new_action_means)
    # entropies = log_probs .* 0.5f0
    return vec(values), log_probs, entropies, st
end

# function get_discrete_logprobs_and_entropy(action_logits::AbstractArray, actions::AbstractArray{<:Int})
#     log_probs_matrix = Lux.logsoftmax(action_logits)
#     probs_matrix = Lux.softmax(action_logits)
#     # batch_size = size(action_logits, 2)
#     log_probs = getindex.(eachcol(log_probs_matrix), vec(actions))
#     entropy = -vec(sum(probs_matrix .* log_probs_matrix, dims=1))
#     return log_probs, entropy
# end

function evaluate_actions(policy::DiscreteActorCriticPolicy, obs::AbstractArray, actions::AbstractArray{<:Int}, ps, st)
    # @info "in evaluate_actions"
    feats, st = extract_features(policy, obs, ps, st)
    new_action_logits, st = get_actions_from_features(policy, feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, feats, ps, st)
    ds = get_distributions(policy, new_action_logits)
    log_probs = logpdf.(ds, eachslice(actions, dims=ndims(actions)))
    entropies = entropy.(ds)
    return vec(values), log_probs, entropies, st
    # # @info "gone through networks"
    # # Fast path: compute log probs and entropy directly using broadcasting
    # log_probs, entropy = get_discrete_logprobs_and_entropy(new_action_logits, actions)

    # return values, log_probs, entropy, st
end

function predict_values(policy::AbstractActorCriticPolicy, obs::AbstractArray, ps, st)
    feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, feats, ps, st)
    return vec(values), st
end
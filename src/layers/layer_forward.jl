# Forward pass implementations for actor-critic layers

function (policy::ContinuousActorCriticLayer)(obs::AbstractArray, ps, st)
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

function (policy::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any})(
        obs::AbstractArray,
        actions::AbstractArray, ps, st
    ) where {N <: AbstractNoise}
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

function (policy::DiscreteActorCriticLayer)(obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    action_logits, st = get_actions_from_features(policy, actor_feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    ds = get_distributions(policy, action_logits)
    #random sample, as this is called during rollout collection
    actions = rand.(ds)
    log_probs = logpdf.(ds, actions)
    return actions, vec(values), log_probs, st
end

# Type-stable feature extraction using dispatch
function extract_features(policy::ContinuousActorCriticLayer{O, A, N, C, SharedFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    feats, feats_st = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(policy::ContinuousActorCriticLayer{O, A, N, C, SeparateFeatures, FE, AH, CH, LS}, obs::AbstractArray, ps, st) where {O, A, N, C, FE, AH, CH, LS}
    actor_feats, actor_feats_st = policy.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = policy.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# For DiscreteActorCriticLayer (3 type parameters)
function extract_features(policy::DiscreteActorCriticLayer{O, A, SharedFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    feats, feats_st = policy.feature_extractor(obs, ps.feature_extractor, st.feature_extractor)
    actor_feats = feats
    critic_feats = feats
    st = merge(st, (; feature_extractor = feats_st))
    return actor_feats, critic_feats, st
end

function extract_features(policy::DiscreteActorCriticLayer{O, A, SeparateFeatures, FE, AH, CH}, obs::AbstractArray, ps, st) where {O, A, FE, AH, CH}
    actor_feats, actor_feats_st = policy.feature_extractor(obs, ps.actor_feature_extractor, st.actor_feature_extractor)
    critic_feats, critic_feats_st = policy.feature_extractor(obs, ps.critic_feature_extractor, st.critic_feature_extractor)
    st = merge(st, (; actor_feature_extractor = actor_feats_st, critic_feature_extractor = critic_feats_st))
    return actor_feats, critic_feats, st
end

# Direct calls to concrete layer fields keep inference intact

function get_actions_from_features(policy::AbstractActorCriticLayer, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    # temprarily remove copy: enzyme cannot handle it
    actions, actor_st = policy.actor_head(feats, ps.actor_head, st.actor_head)
    st = merge(st, (; actor_head = actor_st))
    return actions, st
end

function get_values_from_features(policy::AbstractActorCriticLayer, feats::AbstractArray, ps, st)
    # Use function barrier to isolate type instability
    values, critic_st = policy.critic_head(feats, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

function get_values_from_features(policy::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, feats::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    if ndims(actions) == 1
        actions = batch(actions, action_space(policy))
    end
    inputs = vcat(feats, actions)
    # Use function barrier to isolate type instability
    #TODO: runtime dispatch
    values, critic_st = policy.critic_head(inputs, ps.critic_head, st.critic_head)
    st = merge(st, (; critic_head = critic_st))
    return values, st
end

# For continuous action spaces
# Dispatch on noise type for VCritic policies
function get_distributions(policy::ContinuousActorCriticLayer{<:Any, <:Any, StateIndependantNoise, VCritic, <:Any, <:Any, <:Any, <:Any}, action_means::AbstractArray{T}, log_std::AbstractArray{T}) where {T <: Real}
    batch_dim = ndims(action_means)
    action_means_vec = collect.(eachslice(action_means, dims = batch_dim)) #::Vector{Array{T, ndims(action_means) - 1}}
    #FIXME: runtime dispatch here in DiagGaussian, types not known??
    return DiagGaussian.(action_means_vec, Ref(log_std))
end

function get_distributions(policy::ContinuousActorCriticLayer{<:Any, <:Any, StateDependentNoise, VCritic, <:Any, <:Any, <:Any, <:Any}, action_means::AbstractArray{T}, log_std::AbstractArray{T}) where {T <: Real}
    batch_dim = ndims(action_means)
    @assert size(log_std) == size(action_means) "log_std and action_means have different shapes"
    action_means_vec = collect(eachslice(action_means, dims = batch_dim)) #::Vector{Array{T, ndims(action_means) - 1}}
    log_std_vec = collect(eachslice(log_std, dims = batch_dim)) #::Vector{Array{T, ndims(log_std) - 1}}
    return DiagGaussian.(action_means_vec, log_std_vec)
end

# Dispatch on noise type for QCritic policies
function get_distributions(policy::ContinuousActorCriticLayer{<:Any, <:Any, StateIndependantNoise, QCritic, <:Any, <:Any, <:Any, <:Any}, action_means::AbstractArray{T}, log_std::AbstractArray{T}) where {T <: Real}
    batch_dim = ndims(action_means)
    #FIXME: runtime dispatch here in SquashedDiagGaussian
    #TODO: is collect needed here?
    action_means_vec = collect(eachslice(action_means, dims = batch_dim)) #::Vector{<:AbstractArray{T, batch_dim - 1}}
    return SquashedDiagGaussian.(action_means_vec, Ref(log_std))
end

function get_distributions(policy::ContinuousActorCriticLayer{<:Any, <:Any, StateDependentNoise, QCritic, <:Any, <:Any, <:Any, <:Any}, action_means::AbstractArray{T}, log_std::AbstractArray{T}) where {T <: Real}
    batch_dim = ndims(action_means)
    @assert size(log_std) == size(action_means) "log_std and action_means have different shapes"
    action_means_vec = collect(eachslice(action_means, dims = batch_dim)) #::Vector{<:AbstractArray{T, ndims(action_means) - 1}}
    log_std_vec = collect(eachslice(log_std, dims = batch_dim)) #::Vector{<:AbstractArray{T, ndims(log_std) - 1}}
    return SquashedDiagGaussian.(action_means_vec, log_std_vec)
end

# For discrete action spaces
function get_distributions(policy::DiscreteActorCriticLayer, action_logits::AbstractArray{T}) where {T <: Real}
    # For discrete actions, action_logits are the raw outputs from the network
    # std is not used for discrete actions
    probs = Lux.softmax(action_logits)
    batch_dim = ndims(action_logits)
    start = action_space(policy).start
    probs_vec = collect(eachslice(probs, dims = batch_dim)) #::Vector{<:AbstractArray{T, ndims(probs) - 1}}
    return Categorical.(probs_vec, start)
end

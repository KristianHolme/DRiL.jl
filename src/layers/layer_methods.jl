# High-level actor-critic layer methods

function predict_actions(policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = Random.default_rng())
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

function predict_actions(policy::DiscreteActorCriticLayer, obs::AbstractArray, ps, st; deterministic::Bool = false, rng::AbstractRNG = Random.default_rng())
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

function evaluate_actions(policy::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray{T}, actions::AbstractArray{T}, ps, st) where {T <: Number}
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    new_action_means, st = get_actions_from_features(policy, actor_feats, ps, st) #runtime dispatch
    values, st = get_values_from_features(policy, critic_feats, ps, st) #runtime dispatch
    distributions = get_distributions(policy, new_action_means, ps.log_std) #runtime dispatch
    actions_vec = collect(eachslice(actions, dims = ndims(actions))) #runtime dispatch
    log_probs = logpdf.(distributions, actions_vec)
    entropies = entropy.(distributions) #runtime dispatch
    return evaluate_actions_returns(policy, values, log_probs, entropies, st)
end

function evaluate_actions_returns(::ContinuousActorCriticLayer{<:Any, <:Any, <:Any, QCritic}, values, log_probs, entropies, st)
    return values, log_probs, entropies, st #dont return vec(values) as values is a matrix
end
function evaluate_actions_returns(::ContinuousActorCriticLayer, values, log_probs, entropies, st)
    return vec(values), log_probs, entropies, st
end

function evaluate_actions(policy::DiscreteActorCriticLayer, obs::AbstractArray, actions::AbstractArray{<:Int}, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    new_action_logits, st = get_actions_from_features(policy, actor_feats, ps, st)  # For discrete, these are logits
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    ds = get_distributions(policy, new_action_logits)
    actions_vec = collect(eachslice(actions, dims = ndims(actions))) #::Vector{AbstractArray{T, ndims(actions) - 1}}
    log_probs = logpdf.(ds, actions_vec)
    entropies = entropy.(ds)
    return vec(values), log_probs, entropies, st
end

function predict_values(policy::AbstractActorCriticLayer, obs::AbstractArray, ps, st)
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, critic_feats, ps, st)
    return vec(values), st
end

function predict_values(policy::ContinuousActorCriticLayer{<:Any, <:Any, N, QCritic, <:Any, <:Any, <:Any, <:Any}, obs::AbstractArray, actions::AbstractArray, ps, st) where {N <: AbstractNoise}
    actor_feats, critic_feats, st = extract_features(policy, obs, ps, st)
    values, st = get_values_from_features(policy, critic_feats, actions, ps, st)
    return values, st #dont return vec(values) as this is a matrix
end

#returns vector of actions
function action_log_prob(policy::ContinuousActorCriticLayer, obs::AbstractArray, ps, st; rng::AbstractRNG = Random.default_rng())
    #TODO: fix runtime dispatch here in extract_features
    actor_feats, _, st = extract_features(policy, obs, ps, st)
    action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
    log_std = ps.log_std
    ds = get_distributions(policy, action_means, log_std)
    actions = rand.(rng, ds)
    log_probs = logpdf.(ds, actions)
    # scaled_actions = scale_to_space.(actions, Ref(policy.action_space))
    return actions, log_probs, st
end

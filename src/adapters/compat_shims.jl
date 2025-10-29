# Backward-compatible process_action shims

function process_action(action, space::Box, alg::AbstractAlgorithm)
    @warn "process_action is deprecated; use to_env(action_adapter(alg), action, space) instead" maxlog = 1
    return to_env(action_adapter(alg), action, space)
end

function process_action(action::Integer, space::Discrete, alg::AbstractAlgorithm)
    @warn "process_action is deprecated; use to_env(DiscreteAdapter(), action, space) instead" maxlog = 1
    return to_env(DiscreteAdapter(), action, space)
end

function process_action(action::AbstractArray{<:Integer}, space::Discrete, alg::AbstractAlgorithm)
    @warn "process_action is deprecated; use to_env(DiscreteAdapter(), action, space) instead" maxlog = 1
    return to_env(DiscreteAdapter(), action, space)
end

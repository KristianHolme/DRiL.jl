# Deployment-time policy (actor-only wrapper)

struct DeploymentPolicy{L, AD, S}
    layer::L
    params
    states::S
    action_space
    adapter::AD
end

"""
    extract_policy(agent) -> DeploymentPolicy

Create a lightweight deployment policy from a trained agent.
"""
function extract_policy(agent)
    layer = agent.layer
    ps = agent.train_state.parameters
    st = agent.train_state.states
    as = action_space(layer)
    adapter = agent.action_adapter
    return DeploymentPolicy(layer, ps, st, as, adapter)
end

"""
    predict(dp::DeploymentPolicy, obs_batch; deterministic=true, rng=Random.default_rng())

Return env-space actions for a batch of observations.
"""
function predict(dp::DeploymentPolicy, obs_batch; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    actions, _ = predict_actions(dp.layer, obs_batch, dp.params, dp.states; deterministic = deterministic, rng = rng)
    env_actions = to_env.(Ref(dp.adapter), actions, Ref(dp.action_space))
    return env_actions
end

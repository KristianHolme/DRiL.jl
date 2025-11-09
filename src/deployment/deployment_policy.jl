# Deployment-time policy (actor-only wrapper)

struct DeploymentPolicy{L, AD, S} <: AbstractPolicy
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


function (dp::DeploymentPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    if obs in observation_space(dp.layer)
        obs = batch(obs, observation_space(dp.layer))
    end
    actions, _ = predict_actions(dp.layer, obs, dp.params, dp.states; deterministic = deterministic, rng = rng)
    env_actions = to_env.(Ref(dp.adapter), actions, Ref(dp.action_space))
    return env_actions
end

#TODO: add tests
struct NormalizedDeploymentPolicy{P <: DeploymentPolicy, T <: AbstractFloat} <: AbstractPolicy
    policy::P
    obs_rms::RunningMeanStd{T}
    eps::T
    clip_obs::T
end

function extract_policy(agent, norm_env::NormalizeWrapperEnv)
    policy = extract_policy(agent)
    obs_rms = norm_env.obs_rms
    eps = norm_env.epsilon
    clip_obs = norm_env.clip_obs
    return NormalizedDeploymentPolicy(policy, obs_rms, eps, clip_obs)
end

function (dp::NormalizedDeploymentPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())
    if obs in observation_space(dp.policy.layer)
        obs = batch(obs, observation_space(dp.policy.layer))
    end
    normalize_obs!(obs, dp.obs_rms, dp.eps, dp.clip_obs)
    return dp.policy(obs; deterministic = deterministic, rng = rng)
end

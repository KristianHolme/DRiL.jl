abstract type AbstractAlgorithm end


struct PPO{T<: AbstractFloat} <: AbstractAlgorithm 
    n_steps::Int
    gamma::T
    gae_lambda::T
    clip_range::T
    clip_range_vf::T
    ent_coef::T
    vf_coef::T
    use_sde::Bool
    sde_sample_freq::Int
    target_kl::T
    normalize_advantage::Bool
    stats_window::Int
    verbose::Int
    seed::Union{Int, Nothing}
end

function learn!(policy::ActorCriticPolicy, env::AbstractEnv, alg::PPO, max_steps::Int)
    n_steps = alg.n_steps
    n_envs = env.n_envs
    roll_buffer = RolloutBuffer(policy.observation_space, policy.action_space, alg.gae_lambda, alg.gamma, n_steps, n_envs)
    iterations = max_steps ÷ (n_steps * n_envs)
    for i in 1:iterations
        collect_rollouts!(roll_buffer, policy, env)
    end
end

   
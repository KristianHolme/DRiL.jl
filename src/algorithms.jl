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
end

function learn!(agent::ActorCriticAgent, env::AbstractEnv, alg::PPO{T}, max_steps::Int) where T
    n_steps = alg.n_steps
    n_envs = env.n_envs
    roll_buffer = RolloutBuffer(agent.policy.observation_space, agent.policy.action_space, alg.gae_lambda, alg.gamma, n_steps, n_envs)
    iterations = max_steps ÷ (n_steps * n_envs)
    for i in 1:iterations
        collect_rollouts!(roll_buffer, agent, env)
        data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions, 
                                roll_buffer.advantages, roll_buffer.returns, 
                                roll_buffer.old_logprobs, roll_buffer.values), 
                                batchsize=agent.batch_size, shuffle=true, parallel=true)
        continue_training = true
        #update learning rate?
        for epoch in 1:alg.n_epochs
            for batch_data in data_loader
                alg_loss = (model, ps, st, data) -> loss(alg, model, ps, st, data)
                grads, loss, stats, st = Lux.compute_gradients(agent.ad_type, alg_loss, batch_data, agent.training_state)
                if !isnothing(alg.target_kl) && stats["approx_kl_div"] > T(1.5) * alg.target_kl 
                    continue_training = false
                    break
                end
                Lux.apply_gradients!(agent.train_state, grads)
                #update stats
            end
            continue_training || break
        end
    end
end


function loss(alg::PPO{T}, policy::ActorCriticPolicy, ps, st, batch) where T
    observations, actions, advantages, returns, old_logprobs, values = batch
    
    log_probs, entropy, values, st = Lux.apply(policy, observations, ps, st)

    r = exp.(log_probs - old_logprobs)
    p_loss = -mean(advantages .* min.(r, clamp.(r, 1-alg.clip_range, 1+alg.clip_range)))
    ent_loss = -mean(entropy)
    v_loss = mean((values .- returns).^2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss
    
    stats = Dict()

    return loss, st, stats
end

   
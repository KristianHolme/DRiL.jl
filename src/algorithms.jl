abstract type AbstractAlgorithm end


@kwdef struct PPO{T<: AbstractFloat} <: AbstractAlgorithm 
    gamma::T = 0.99f0
    gae_lambda::T = 0.95f0
    clip_range::T = 0.2f0
    clip_range_vf::Union{T, Nothing} = nothing
    ent_coef::T = 0.0f0
    vf_coef::T = 0.5f0
    max_grad_norm::T = 0.5f0
    target_kl::Union{T, Nothing} = nothing
    normalize_advantage::Bool = true
end
function PPO(;kwargs...)
    return PPO{Float32}(;kwargs...)
end

function learn!(agent::ActorCriticAgent, env::AbstractEnv, alg::PPO{T}; max_steps::Int, ad_type::Lux.Training.AbstractADType=AutoEnzyme()) where T
    n_steps = agent.n_steps
    n_envs = env.n_envs
    roll_buffer = RolloutBuffer(observation_space(env), action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    iterations = max_steps ÷ (n_steps * n_envs)
    total_steps = iterations * n_steps * n_envs

    progress_meter = Progress(total_steps, desc="Training...")

    train_state = agent.train_state

    total_entropy_losses = Float32[]
    total_policy_losses = Float32[]
    total_value_losses = Float32[]
    total_approx_kl_divs = Float32[]
    total_clip_fractions = Float32[]
    total_losses = Float32[]
    total_explained_variances = Float32[]
    total_fps = Float32[]
    for i in 1:iterations
        fps = collect_rollouts!(roll_buffer, agent, env, progress_meter)
        push!(total_fps, fps)
        data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions, 
                                roll_buffer.advantages, roll_buffer.returns, 
                                roll_buffer.old_logprobs, roll_buffer.values), 
                                batchsize=agent.batch_size, shuffle=true, parallel=true)
        continue_training = true
        entropy_losses = Float32[]
        policy_losses = Float32[]
        value_losses = Float32[]
        approx_kl_divs = Float32[]
        clip_fractions = Float32[]
        #update learning rate?
        for epoch in 1:alg.epochs
            for batch_data in data_loader
                alg_loss = (model, ps, st, data) -> loss(alg, model, ps, st, data)
                grads, loss_val, stats, train_state = Lux.compute_gradients(ad_type, alg_loss, batch_data, train_state)
                
                # KL divergence check
                if !isnothing(alg.target_kl) && stats["approx_kl_div"] > T(1.5) * alg.target_kl 
                    continue_training = false
                    break
                end
                Lux.apply_gradients!(train_state, grads)
                push!(entropy_losses, stats["entropy_loss"])
                push!(policy_losses, stats["policy_loss"])
                push!(value_losses, stats["value_loss"])
                push!(approx_kl_divs, stats["approx_kl_div"])
                push!(clip_fractions, stats["clip_fraction"])
                push!(total_losses, loss_val)
            end
            if !continue_training
                @info "Early stopping at epoch $epoch in iteration $i, due to KL divergence"
                break
            end
        end

        explained_variance = 1 - var(roll_buffer.values .- roll_buffer.returns) / var(roll_buffer.returns)
        push!(total_explained_variances, explained_variance)
        push!(total_entropy_losses, mean(entropy_losses))
        push!(total_policy_losses, mean(policy_losses))
        push!(total_value_losses, mean(value_losses))
        push!(total_approx_kl_divs, mean(approx_kl_divs))
        push!(total_clip_fractions, mean(clip_fractions))
        push!(total_losses, mean(total_losses))
        if agent.verbose > 1
            ProgressMeter.update!(progress_meter; showvalues=[
                ("explained_variance", explained_variance),
                ("entropy_loss", total_entropy_losses[i]),
                ("policy_loss", total_policy_losses[i]),
                ("value_loss", total_value_losses[i]),
                ("approx_kl_div", total_approx_kl_divs[i]),
                ("clip_fraction", total_clip_fractions[i]),
                ("loss", total_losses[i]),
                ("fps", total_fps[i])
            ])
        end
    end
end

function normalize!(advantages::Vector{T}) where T
    mean_adv = mean(advantages)
    std_adv = std(advantages)
    advantages .= (advantages .- mean_adv) ./ std_adv
    nothing
end

function clip_range!(values::Vector{T}, old_values::Vector{T}, clip_range::T) where T
    values .= old_values .+ clamp.(values .- old_values, -clip_range, clip_range)
    nothing
end

function loss(alg::PPO{T}, policy::ActorCriticPolicy, ps, st, batch_data) where T
    observations, actions, advantages, returns, old_logprobs, old_values = batch_data

    alg.normalize_advantage && normalize!(advantages)

    values, log_probs, entropy, st = evaluate_actions(policy, observations, actions, ps, st)

    !isnothing(alg.clip_range_vf) && clip_range!(values, old_values, alg.clip_range_vf)

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1-alg.clip_range, 1+alg.clip_range)
    p_loss = -mean(advantages .* min.(r, ratio_clipped))
    ent_loss = -mean(entropy)
    v_loss = mean((values .- returns).^2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss
    
    # Calculate statistics
    clip_fraction = mean(r .!= ratio_clipped)

    #approx kl div
    log_ratio = log_probs - old_logprobs
    approx_kl_div = mean(exp.(log_ratio) - 1 - log_ratio)
    
    stats = Dict(
        "policy_loss" => p_loss,
        "value_loss" => v_loss,
        "entropy_loss" => ent_loss,
        "clip_fraction" => clip_fraction,
        "approx_kl_div" => approx_kl_div,
    )

    return loss, st, stats
end

   
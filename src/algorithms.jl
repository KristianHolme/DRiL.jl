abstract type AbstractAlgorithm end


@kwdef struct PPO{T<:AbstractFloat} <: AbstractAlgorithm
    gamma::T = 0.99f0
    gae_lambda::T = 0.95f0
    clip_range::T = 0.2f0
    clip_range_vf::Union{T,Nothing} = nothing
    ent_coef::T = 0.0f0
    vf_coef::T = 0.5f0
    max_grad_norm::T = 0.5f0
    target_kl::Union{T,Nothing} = nothing
    normalize_advantage::Bool = true
end

# function PPO(; T::Type{<:AbstractFloat}=Float32, kwargs...)
#     return PPO{T}(; kwargs...)
# end

function learn!(agent::ActorCriticAgent, env::AbstractEnv, alg::PPO{T}; max_steps::Int, ad_type::Lux.Training.AbstractADType=AutoZygote()) where T
    n_steps = agent.n_steps
    n_envs = env.n_envs
    roll_buffer = RolloutBuffer(observation_space(env), action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    iterations = max_steps ÷ (n_steps * n_envs)
    total_steps = iterations * n_steps * n_envs

    @info "Training with total_steps: $total_steps, iterations: $iterations, n_steps: $n_steps, n_envs: $n_envs"
    progress_meter = Progress(total_steps, desc="Training...",
        showspeed=true, enabled=agent.verbose > 0
    )

    train_state = agent.train_state

    total_entropy_losses = Float32[]
    total_entropy = Float32[]
    total_policy_losses = Float32[]
    total_value_losses = Float32[]
    total_approx_kl_divs = Float32[]
    total_clip_fractions = Float32[]
    total_losses = Float32[]
    total_explained_variances = Float32[]
    total_fps = Float32[]
    total_grad_norms = Float32[]

    for i in 1:iterations
        fps = collect_rollouts!(roll_buffer, agent, env, progress_meter)
        push!(total_fps, fps)
        add_step!(agent, n_steps * n_envs)
        if !isnothing(agent.logger)
            set_step!(agent.logger, steps_taken(agent))
            log_value(agent.logger, "env/fps", fps)
        end
        data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values),
            batchsize=agent.batch_size, shuffle=true, parallel=true)
        continue_training = true
        entropy_losses = Float32[]
        entropy = Float32[]
        policy_losses = Float32[]
        value_losses = Float32[]
        losses = Float32[]
        approx_kl_divs = Float32[]
        clip_fractions = Float32[]
        grad_norms = Float32[]
        for epoch in 1:agent.epochs
            for (i_batch, batch_data) in enumerate(data_loader)
                alg_loss = (model, ps, st, data) -> loss(alg, model, ps, st, data)
                grads, loss_val, stats, train_state = Lux.Training.compute_gradients(ad_type, alg_loss, batch_data, train_state)

                if epoch == 1 && i_batch == 1
                    mean_ratio = stats["ratio"]
                    @assert mean_ratio ≈ one(mean_ratio) "ratios is not 1.0, iter $i, epoch $epoch, batch $i_batch, $mean_ratio"
                end
                @assert !any(isnan, grads.log_std) "log_std gradient is nan, iter $i, epoch $epoch, batch $i_batch"

                # Calculate and store gradient norm for the batch
                flat_grads, restruct_func = Optimisers.destructure(grads)
                current_grad_norm = norm(flat_grads)
                push!(grad_norms, current_grad_norm)
                if !isnothing(alg.max_grad_norm) && current_grad_norm > alg.max_grad_norm
                    flat_grads = flat_grads .* alg.max_grad_norm ./ current_grad_norm
                    @assert norm(flat_grads) <= alg.max_grad_norm "gradient norm is greater than max_grad_norm, iter $i, epoch $epoch, batch $i_batch"
                    grads = restruct_func(flat_grads)
                end
                # @info grads
                # KL divergence check
                if !isnothing(alg.target_kl) && stats["approx_kl_div"] > T(1.5) * alg.target_kl
                    continue_training = false
                    break
                end
                Lux.Training.apply_gradients!(train_state, grads)
                add_gradient_update!(agent)
                push!(entropy, stats["entropy"])
                push!(entropy_losses, stats["entropy_loss"])
                push!(policy_losses, stats["policy_loss"])
                push!(value_losses, stats["value_loss"])
                push!(approx_kl_divs, stats["approx_kl_div"])
                push!(clip_fractions, stats["clip_fraction"])
                push!(losses, loss_val)
            end
            if !continue_training
                @info "Early stopping at epoch $epoch in iteration $i, due to KL divergence"
                break
            end
        end

        explained_variance = 1 - var(roll_buffer.values .- roll_buffer.returns) / var(roll_buffer.returns)
        push!(total_explained_variances, explained_variance)
        push!(total_entropy_losses, mean(entropy_losses))
        push!(total_entropy, mean(entropy))
        push!(total_policy_losses, mean(policy_losses))
        push!(total_value_losses, mean(value_losses))
        push!(total_approx_kl_divs, mean(approx_kl_divs))
        push!(total_clip_fractions, mean(clip_fractions))
        push!(total_losses, mean(losses))
        push!(total_grad_norms, mean(grad_norms))
        if agent.verbose > 1
            ProgressMeter.update!(progress_meter; showvalues=[
                ("explained_variance", explained_variance),
                ("entropy_loss", total_entropy_losses[i]),
                ("entropy", total_entropy[i]),
                ("policy_loss", total_policy_losses[i]),
                ("value_loss", total_value_losses[i]),
                ("approx_kl_div", total_approx_kl_divs[i]),
                ("clip_fraction", total_clip_fractions[i]),
                ("loss", total_losses[i]),
                ("fps", total_fps[i]),
                ("grad_norm", total_grad_norms[i])
            ])
        end
        if !isnothing(agent.logger)
            log_value(agent.logger, "train/entropy_loss", total_entropy_losses[i])
            log_value(agent.logger, "train/entropy", total_entropy[i])
            log_value(agent.logger, "train/explained_variance", explained_variance)
            log_value(agent.logger, "train/policy_loss", total_policy_losses[i])
            log_value(agent.logger, "train/value_loss", total_value_losses[i])
            log_value(agent.logger, "train/approx_kl_div", total_approx_kl_divs[i])
            log_value(agent.logger, "train/clip_fraction", total_clip_fractions[i])
            log_value(agent.logger, "train/loss", total_losses[i])
            log_value(agent.logger, "train/grad_norm", total_grad_norms[i])
            if haskey(agent.train_state.parameters, :log_std)
                log_value(agent.logger, "train/std", mean(exp.(agent.train_state.parameters[:log_std])))
            end
        end
    end
end

function normalize(advantages::Vector{T}) where T
    mean_adv = mean(advantages)
    std_adv = std(advantages)
    epsilon = T(1e-8)
    norm_advantages = (advantages .- mean_adv) ./ (std_adv + epsilon)
    return norm_advantages
end

function clip_range!(values::Vector{T}, old_values::Vector{T}, clip_range::T) where T
    for i in eachindex(values)
        diff = values[i] - old_values[i]
        clipped_diff = clamp(diff, -clip_range, clip_range)
        values[i] = old_values[i] + clipped_diff
    end
    nothing
end

function clip_range(old_values::Vector{T}, values::Vector{T}, clip_range::T) where T
    return old_values .+ clamp.(values .- old_values, -clip_range, clip_range)
end

function loss(alg::PPO{T}, policy::ActorCriticPolicy, ps, st, batch_data) where T
    observations, actions, advantages, returns, old_logprobs, old_values = batch_data

    advantages = @ignore_derivatives alg.normalize_advantage ? normalize(advantages) : advantages

    values, log_probs, entropy, st = evaluate_actions(policy, observations, actions, ps, st)

    log_probs = vec(log_probs)
    values = vec(values)
    entropy = vec(entropy)

    values = !isnothing(alg.clip_range_vf) ? clip_range(old_values, values, alg.clip_range_vf) : values

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1 - alg.clip_range, 1 + alg.clip_range)
    p_loss = -mean(advantages .* min.(r, ratio_clipped))
    ent_loss = -mean(entropy)

    # @info "values: $values"
    # @info "returns: $returns"
    v_loss = mean((values .- returns) .^ 2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss

    # Calculate statistics
    clip_fraction = mean(r .!= ratio_clipped)
    #approx kl div
    log_ratio = log_probs - old_logprobs
    approx_kl_div = mean(exp.(log_ratio) .- 1 .- log_ratio)

    stats = Dict(
        "policy_loss" => p_loss,
        "value_loss" => v_loss,
        "entropy_loss" => ent_loss,
        "clip_fraction" => clip_fraction,
        "approx_kl_div" => approx_kl_div,
        "entropy" => mean(entropy),
        "ratio" => mean(r)
    )

    return loss, st, stats
end


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
#TODO make parameters n_steps, batch_size, epochs, max_steps kwargs, default to values from agent
function learn!(agent::ActorCriticAgent, env::AbstractParallellEnv, alg::PPO{T}, ad_type::Lux.Training.AbstractADType=AutoZygote(); max_steps::Int) where T
    n_steps = agent.n_steps
    n_envs = number_of_envs(env)
    roll_buffer = RolloutBuffer(observation_space(env), action_space(env),
        alg.gae_lambda, alg.gamma, n_steps, n_envs)

    iterations = max_steps ÷ (n_steps * n_envs)
    total_steps = iterations * n_steps * n_envs

    agent.verbose > 0 && @info "Training with total_steps: $total_steps, 
        iterations: $iterations, n_steps: $n_steps, n_envs: $n_envs"

    progress_meter = Progress(total_steps, desc="Training...",
        showspeed=true, enabled=agent.verbose > 0
    )

    train_state = agent.train_state

    total_entropy_losses = Float32[]
    learning_rates = Float32[]
    total_policy_losses = Float32[]
    total_value_losses = Float32[]
    total_approx_kl_divs = Float32[]
    total_clip_fractions = Float32[]
    total_losses = Float32[]
    total_explained_variances = Float32[]
    total_fps = Float32[]
    total_grad_norms = Float32[]

    for i in 1:iterations
        learning_rate = agent.learning_rate
        Optimisers.adjust!(agent.train_state, learning_rate)
        push!(learning_rates, learning_rate)

        fps = collect_rollouts!(roll_buffer, agent, env, progress_meter)
        push!(total_fps, fps)
        add_step!(agent, n_steps * n_envs)
        if !isnothing(agent.logger)
            set_step!(agent.logger, steps_taken(agent))
            log_value(agent.logger, "env/fps", fps)
            log_stats(env, agent.logger)
        end
        data_loader = DataLoader((roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values),
            batchsize=agent.batch_size, shuffle=true, parallel=true, rng=agent.rng)
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
                    mean_ratio ≈ one(mean_ratio) || @warn "ratios is not 1.0, iter $i, epoch $epoch, batch $i_batch, $mean_ratio"
                end
                @assert !any(isnan, grads) "gradient contains nan, iter $i, epoch $epoch, batch $i_batch"
                @assert !any(isinf, grads) "gradient not finite, iter $i, epoch $epoch, batch $i_batch"

                current_grad_norm = norm(grads)
                # @info "actor grad norm: $(norm(grads.actor_head))"
                if norm(grads.actor_head) < 1e-3
                    @info "actor grad" grads.actor_head
                end
                # @info "critic grad norm: $(norm(grads.critic_head))"
                # @info "log_std grad norm: $(norm(grads.log_std))"
                push!(grad_norms, current_grad_norm)

                if !isnothing(alg.max_grad_norm) && current_grad_norm > alg.max_grad_norm
                    grads = grads .* alg.max_grad_norm ./ current_grad_norm
                    clipped_grads_norm = norm(grads)
                    @assert clipped_grads_norm < alg.max_grad_norm ||
                            clipped_grads_norm ≈ alg.max_grad_norm "gradient norm 
                            ($(clipped_grads_norm)) is greater than
                            max_grad_norm ($(alg.max_grad_norm)), iter $i, epoch $epoch, batch $i_batch"
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
                ("policy_loss", total_policy_losses[i]),
                ("value_loss", total_value_losses[i]),
                ("approx_kl_div", total_approx_kl_divs[i]),
                ("clip_fraction", total_clip_fractions[i]),
                ("loss", total_losses[i]),
                ("fps", total_fps[i]),
                ("grad_norm", total_grad_norms[i]),
                ("learning_rate", learning_rate)
            ])
        end
        if !isnothing(agent.logger)
            log_value(agent.logger, "train/entropy_loss", total_entropy_losses[i])
            log_value(agent.logger, "train/explained_variance", explained_variance)
            log_value(agent.logger, "train/policy_loss", total_policy_losses[i])
            log_value(agent.logger, "train/value_loss", total_value_losses[i])
            log_value(agent.logger, "train/approx_kl_div", total_approx_kl_divs[i])
            log_value(agent.logger, "train/clip_fraction", total_clip_fractions[i])
            log_value(agent.logger, "train/loss", total_losses[i])
            log_value(agent.logger, "train/grad_norm", total_grad_norms[i])
            log_value(agent.logger, "train/learning_rate", learning_rate)
            if haskey(agent.train_state.parameters, :log_std)
                log_value(agent.logger, "train/std", mean(exp.(agent.train_state.parameters[:log_std])))
            end
        end
    end
    learn_stats = Dict(
        "entropy_losses" => total_entropy_losses,
        "policy_losses" => total_policy_losses,
        "value_losses" => total_value_losses,
        "approx_kl_divs" => total_approx_kl_divs,
        "clip_fractions" => total_clip_fractions,
        "losses" => total_losses,
        "explained_variances" => total_explained_variances,
        "fps" => total_fps,
        "grad_norms" => total_grad_norms,
        "learning_rates" => learning_rates
    )
    return learn_stats
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

function loss(alg::PPO{T}, policy::AbstractActorCriticPolicy, ps, st, batch_data) where T
    observations, actions, advantages, returns, old_logprobs, old_values = @ignore_derivatives batch_data

    advantages = @ignore_derivatives alg.normalize_advantage ? normalize(advantages) : advantages
    values, log_probs, entropy, st = evaluate_actions(policy, observations, actions, ps, st)

    values = !isnothing(alg.clip_range_vf) ? clip_range(old_values, values, alg.clip_range_vf) : values

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1 - alg.clip_range, 1 + alg.clip_range)
    p_loss = -mean(min.(r .* advantages, ratio_clipped .* advantages))
    ent_loss = -mean(entropy)

    # @info "values: $values"
    # @info "returns: $returns"
    v_loss = mean((values .- returns) .^ 2)
    loss = p_loss + alg.ent_coef * ent_loss + alg.vf_coef * v_loss

    stats = Dict()
    @ignore_derivatives begin
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
    end

    return loss, st, stats
end


@kwdef struct PPO{T <: AbstractFloat} <: OnPolicyAlgorithm
    gamma::T = 0.99f0
    gae_lambda::T = 0.95f0
    clip_range::T = 0.2f0
    clip_range_vf::Union{T, Nothing} = nothing
    ent_coef::T = 0.0f0
    vf_coef::T = 0.5f0
    max_grad_norm::T = 0.5f0
    target_kl::Union{T, Nothing} = nothing
    normalize_advantage::Bool = true
    # Agent parameters moved from ActorCriticAgent
    n_steps::Int = 2048
    batch_size::Int = 64
    epochs::Int = 10
    learning_rate::T = 3.0f-4
end

function ActorCriticAgent(
        policy::AbstractActorCriticPolicy, alg::PPO;
        optimizer_type::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
        stats_window::Int = 100, #TODO not used
        verbose::Int = 1,
        log_dir::Union{Nothing, String} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )

    optimizer = make_optimizer(optimizer_type, alg)
    ps, st = Lux.setup(rng, policy)
    # @show ps.log_std
    if !isnothing(log_dir)
        logger = TBLogger(log_dir, tb_increment)
    else
        logger = nothing
    end
    train_state = Lux.Training.TrainState(policy, ps, st, optimizer)
    return ActorCriticAgent(
        policy, train_state, optimizer_type, stats_window,
        logger, verbose, rng, AgentStats(0, 0)
    )
end

function make_optimizer(optimizer_type::Type{<:Optimisers.Adam}, alg::PPO)
    return optimizer_type(eta = alg.learning_rate, epsilon = 1.0f-5)
end

function load_policy_params_and_state!(agent::ActorCriticAgent, alg::PPO, path::AbstractString; suffix::String = ".jld2")
    file_path = endswith(path, suffix) ? path : path * suffix
    @info "Loading policy, parameters, and state from $file_path"
    data = load(file_path)
    new_policy = data["policy"]
    new_parameters = data["parameters"]
    new_states = data["states"]
    new_optimizer = make_optimizer(agent.optimizer_type, alg)
    new_train_state = Lux.Training.TrainState(new_policy, new_parameters, new_states, new_optimizer)
    agent.policy = new_policy
    agent.train_state = new_train_state
    return agent
end

#TODO make parameters n_steps, batch_size, epochs, max_steps kwargs, default to values from agent
#TODO refactor, separate out learnig loop and logging
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; ad_type::Lux.Training.AbstractADType = AutoZygote(), callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    roll_buffer = RolloutBuffer(
        observation_space(env), action_space(env),
        alg.gae_lambda, alg.gamma, n_steps, n_envs
    )

    iterations = max_steps ÷ (n_steps * n_envs)
    total_steps = iterations * n_steps * n_envs

    agent.verbose > 0 && @info "Training with total_steps: $total_steps, 
    iterations: $iterations, n_steps: $n_steps, n_envs: $n_envs"

    progress_meter = Progress(
        total_steps, desc = "Training...",
        showspeed = true, enabled = agent.verbose > 0
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

    if !isnothing(callbacks)
        if !all(c -> on_training_start(c, Base.@locals), callbacks)
            @warn "Training stopped due to callback failure"
            return nothing
        end
    end

    for i in 1:iterations
        learning_rate = alg.learning_rate
        Optimisers.adjust!(agent.train_state, learning_rate)
        push!(learning_rates, learning_rate)

        if !isnothing(callbacks)
            if !all(c -> on_rollout_start(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return nothing
            end
        end
        fps, success = collect_rollout!(roll_buffer, agent, alg, env, progress_meter; callbacks = callbacks)
        if !success
            @warn "Training stopped due to callback failure"
            return nothing
        end
        push!(total_fps, fps)
        add_step!(agent, n_steps * n_envs)
        if !isnothing(agent.logger)
            logger = agent.logger::TensorBoardLogger.TBLogger
            set_step!(logger, steps_taken(agent))
            log_value(logger, "env/fps", fps)
            log_stats(env, logger)
        end

        if !isnothing(callbacks)
            if !all(c -> on_rollout_end(c, Base.@locals), callbacks)
                @warn "Training stopped due to callback failure"
                return nothing
            end
        end

        data_loader = DataLoader(
            (
                roll_buffer.observations, roll_buffer.actions,
                roll_buffer.advantages, roll_buffer.returns,
                roll_buffer.logprobs, roll_buffer.values,
            ),
            batchsize = alg.batch_size, shuffle = true, parallel = true, rng = agent.rng
        )
        continue_training = true
        entropy_losses = Float32[]
        entropy = Float32[]
        policy_losses = Float32[]
        value_losses = Float32[]
        losses = Float32[]
        approx_kl_divs = Float32[]
        clip_fractions = Float32[]
        grad_norms = Float32[]
        for epoch in 1:alg.epochs
            for (i_batch, batch_data) in enumerate(data_loader)
                grads, loss_val, stats, train_state = Lux.Training.compute_gradients(ad_type, alg, batch_data, train_state)

                if epoch == 1 && i_batch == 1
                    mean_ratio = stats["ratio"]
                    isapprox(mean_ratio - one(mean_ratio), zero(mean_ratio), atol = eps(typeof(mean_ratio))) || @warn "ratios is not 1.0, iter $i, epoch $epoch, batch $i_batch, $mean_ratio"
                end
                @assert !any(isnan, grads) "gradient contains nan, iter $i, epoch $epoch, batch $i_batch"
                @assert !any(isinf, grads) "gradient not finite, iter $i, epoch $epoch, batch $i_batch"

                current_grad_norm = norm(grads)
                # @info "actor grad norm: $(norm(grads.actor_head))"
                if norm(grads.actor_head) < 1.0e-3
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
            ProgressMeter.update!(
                progress_meter; showvalues = [
                    ("explained_variance", explained_variance),
                    ("entropy_loss", total_entropy_losses[i]),
                    ("policy_loss", total_policy_losses[i]),
                    ("value_loss", total_value_losses[i]),
                    ("approx_kl_div", total_approx_kl_divs[i]),
                    ("clip_fraction", total_clip_fractions[i]),
                    ("loss", total_losses[i]),
                    ("fps", total_fps[i]),
                    ("grad_norm", total_grad_norms[i]),
                    ("learning_rate", learning_rate),
                ]
            )
        end
        if !isnothing(agent.logger)
            logger = agent.logger::TensorBoardLogger.TBLogger #to satisfy JET
            log_value(logger, "train/entropy_loss", total_entropy_losses[i])
            log_value(logger, "train/explained_variance", explained_variance)
            log_value(logger, "train/policy_loss", total_policy_losses[i])
            log_value(logger, "train/value_loss", total_value_losses[i])
            log_value(logger, "train/approx_kl_div", total_approx_kl_divs[i])
            log_value(logger, "train/clip_fraction", total_clip_fractions[i])
            log_value(logger, "train/loss", total_losses[i])
            log_value(logger, "train/grad_norm", total_grad_norms[i])
            log_value(logger, "train/learning_rate", learning_rate)
            if haskey(train_state.parameters, :log_std)
                log_value(logger, "train/std", mean(exp.(train_state.parameters[:log_std])))
            end
        end
    end
    agent.train_state = train_state

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
    if !isnothing(callbacks)
        if !all(c -> on_training_end(c, Base.@locals), callbacks)
            @warn "Training stopped due to callback failure"
            return nothing
        end
    end
    return learn_stats
end

function normalize(advantages::Vector{T}) where {T}
    mean_adv = mean(advantages)
    std_adv = std(advantages)
    epsilon = T(1.0e-8)
    norm_advantages = (advantages .- mean_adv) ./ (std_adv + epsilon)
    return norm_advantages
end

function clip_range!(values::Vector{T}, old_values::Vector{T}, clip_range::T) where {T}
    for i in eachindex(values)
        diff = values[i] - old_values[i]
        clipped_diff = clamp(diff, -clip_range, clip_range)
        values[i] = old_values[i] + clipped_diff
    end
    return nothing
end

function clip_range(old_values::Vector{T}, values::Vector{T}, clip_range::T) where {T}
    return old_values .+ clamp(values .- old_values, -clip_range, clip_range)
end

function (alg::PPO{T})(policy::AbstractActorCriticPolicy, ps, st, batch_data) where {T}
    observations = batch_data[1]
    actions = batch_data[2]
    advantages = batch_data[3]
    returns = batch_data[4]
    old_logprobs = batch_data[5]
    old_values = batch_data[6]

    advantages = @ignore_derivatives alg.normalize_advantage ? normalize(advantages) : advantages

    values, log_probs, entropy, st = evaluate_actions(policy, observations, actions, ps, st)
    values = !isnothing(alg.clip_range_vf) ? clip_range(old_values, values, alg.clip_range_vf) : values

    r = exp.(log_probs - old_logprobs)
    ratio_clipped = clamp.(r, 1 - alg.clip_range, 1 + alg.clip_range)
    p_loss = -mean(min.(r .* advantages, ratio_clipped .* advantages))
    ent_loss = -mean(entropy)

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


# Helper function to process actions: ensure correct type and clipping for Box
#TODO performance
function process_action(action, action_space::Box{T}, ::PPO) where {T}
    # First check if type conversion is needed
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    # Then clip to bounds element-wise
    action = clamp.(action, action_space.low, action_space.high)
    return action
end

# Helper function to process actions: convert from 1-based indexing to action space range
function process_action(action::Integer, action_space::Discrete, ::PPO)
    # Make sure its in valid range
    @assert action_space.start ≤ action ≤ action_space.start + action_space.n - 1
    return action
end

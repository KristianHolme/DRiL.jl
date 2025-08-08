@kwdef struct SAC{T<:AbstractFloat,E<:AbstractEntropyCoefficient} <: OffPolicyAlgorithm
    learning_rate::T = 3f-4 #learning rate
    buffer_capacity::Int = 1_000_000
    start_steps::Int = 100 # how many steps to collect with random actions before first gradient update
    batch_size::Int = 256
    tau::T = 0.005f0 #soft update rate
    gamma::T = 0.99f0 #discount
    train_freq::Int = 1
    gradient_steps::Int = 1 # number of gradient updates per train_freq steps, -1 to do as many updates as steps (train_freq)
    ent_coef::E = AutoEntropyCoefficient()
    target_update_interval::Int = 1 # how often to update the target networks
end

# Helper function to calculate target entropy for automatic entropy coefficient
function get_target_entropy(ent_coef::AutoEntropyCoefficient{T}, action_space) where T
    if ent_coef.target isa AutoEntropyTarget
        # For continuous action spaces, target entropy is typically -dim(action_space)
        return -T(prod(size(action_space)))
    elseif ent_coef.target isa FixedEntropyTarget
        return ent_coef.target.target
    else
        error("Unknown entropy target type: $(typeof(ent_coef.target))")
    end
end

function get_gradient_steps(alg::SAC, train_freq::Int=alg.train_freq)
    if alg.gradient_steps == -1
        return train_freq
    else
        return alg.gradient_steps
    end
end

get_target_entropy(ent_coef::FixedEntropyCoefficient, action_space) = nothing

function sac_ent_coef_loss(::SAC,
    policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, ps, st, data;
    rng::AbstractRNG=Random.default_rng()
)
    log_ent_coef = ps.log_ent_coef[1]
    policy_ps = data.policy_ps
    policy_st = data.policy_st
    _, log_probs_pi, policy_st = action_log_prob(policy, data.observations, policy_ps, policy_st; rng)
    target_entropy = data.target_entropy
    loss = -(log_ent_coef * @ignore_derivatives(log_probs_pi .+ target_entropy |> mean))
    return loss, st, Dict()
end

function sac_actor_loss(::SAC, policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, ps, st, data;
    rng::AbstractRNG=Random.default_rng()
)
    obs = data.observations
    ent_coef = data.log_ent_coef[1] |> exp
    actions_pi, log_probs_pi, st = action_log_prob(policy, obs, ps, st; rng)
    q_values, st = predict_values(policy, obs, actions_pi, ps, st)
    min_q_values = minimum(q_values, dims=1) |> vec
    loss = mean(ent_coef .* log_probs_pi - min_q_values)
    return loss, st, Dict()
end

function sac_critic_loss(alg::SAC, policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, ps, st, data;
    rng::AbstractRNG=Random.default_rng()
)
    obs, actions, rewards, terminated, _, next_obs = data.observations, data.actions, data.rewards, data.terminated, data.truncated, data.next_observations
    gamma = alg.gamma
    ent_coef = data.log_ent_coef[1] |> exp
    target_ps = data.target_ps
    target_st = data.target_st

    # Current Q-values
    current_q_values, new_st = predict_values(policy, obs, actions, ps, st)

    # Target Q-values (no gradients)
    obs_dims = ndims(obs)
    next_obs = selectdim(next_obs, obs_dims, .!terminated)
    @assert !any(isnan, next_obs) "Next observations contain NaNs"
    target_q_values = @ignore_derivatives begin
        next_actions, next_log_probs, st = action_log_prob(policy, next_obs, ps, st; rng)
        #replace critic ps and st with target
        ps_with_target = merge(ps, target_ps)
        st_with_target = merge(st, target_st)
        next_q_vals, _ = predict_values(policy, next_obs, next_actions, ps_with_target, st_with_target)
        min_next_q = minimum(next_q_vals, dims=1) |> vec

        # Add entropy term
        next_q_vals_with_entropy = min_next_q .- ent_coef .* next_log_probs
        target_q_vals = rewards
        # Bellman target
        target_q_vals[.!terminated] .+= gamma .* next_q_vals_with_entropy
        target_q_vals
    end

    # Critic loss (sum over all Q-networks)
    T = eltype(current_q_values)
    critic_loss = T(0.5) * sum(mean((current_q .- target_q_values) .^ 2) for current_q in eachrow(current_q_values))

    stats = Dict("mean_q_values" => mean(current_q_values))
    return critic_loss, new_st, stats
end

# Callable functions for SAC algorithm - needed for Lux.Training.compute_gradients
function (alg::SAC)(::ContinuousActorCriticPolicy, ps, st, batch_data)
    # This is the combined loss function for all networks
    # In practice, we'll compute separate losses for actor, critic, and entropy coefficient
    error("SAC algorithm object should not be called directly. Use specific loss functions instead.")
end

# SACAgent and related helper functions moved from agents.jl
struct SACAgent <: AbstractAgent
    policy::ContinuousActorCriticPolicy
    train_state::Lux.Training.TrainState
    Q_target_parameters::ComponentArray
    Q_target_states::NamedTuple
    ent_train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing,TensorBoardLogger.TBLogger}
    verbose::Int
    rng::AbstractRNG
    stats::AgentStats
end

function SACAgent(policy::ContinuousActorCriticPolicy, alg::SAC;
    optimizer_type::Type{<:Optimisers.AbstractRule}=Optimisers.Adam,
    log_dir::Union{Nothing,String}=nothing,
    stats_window::Int=100,
    rng::AbstractRNG=Random.default_rng(),
    verbose::Int=1
)
    ps, st = Lux.setup(rng, policy)
    if !isnothing(log_dir)
        logger = TBLogger(log_dir)
    else
        logger = nothing
    end
    optimizer = make_optimizer(optimizer_type, alg)
    train_state = Lux.Training.TrainState(policy, ps, st, optimizer)
    Q_target_parameters = copy_critic_parameters(policy, ps)
    Q_target_states = copy_critic_states(policy, st)

    # Always initialize entropy coefficient train state
    ent_coef_params = init_entropy_coefficient(alg.ent_coef)
    ent_optimizer = make_optimizer(optimizer_type, alg)
    ent_train_state = Lux.Training.TrainState(policy, ent_coef_params, NamedTuple(), ent_optimizer)

    return SACAgent(policy, train_state, Q_target_parameters, Q_target_states,
        ent_train_state, optimizer_type, stats_window, logger, verbose, rng,
        AgentStats(0, 0)
    )
end

add_step!(agent::SACAgent, steps::Int=1) = add_step!(agent.stats, steps)
add_gradient_update!(agent::SACAgent, updates::Int=1) = add_gradient_update!(agent.stats, updates)
steps_taken(agent::SACAgent) = steps_taken(agent.stats)
gradient_updates(agent::SACAgent) = gradient_updates(agent.stats)

function copy_critic_parameters(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,QCritic}, ps::ComponentArray) where N<:AbstractNoise
    if policy.shared_features
        ComponentArray((feature_extractor=copy(ps.feature_extractor), critic_head=copy(ps.critic_head)))
    else
        ComponentArray((critic_feature_extractor=copy(ps.critic_feature_extractor), critic_head=copy(ps.critic_head)))
    end
end

function copy_critic_states(policy::ContinuousActorCriticPolicy{<:Any,<:Any,N,QCritic}, st::NamedTuple) where N<:AbstractNoise
    if policy.shared_features
        (feature_extractor=deepcopy(st.feature_extractor), critic_head=deepcopy(st.critic_head))
    else
        (critic_feature_extractor=deepcopy(st.critic_feature_extractor), critic_head=deepcopy(st.critic_head))
    end
end

function init_entropy_coefficient(entropy_coefficient::FixedEntropyCoefficient)
    ComponentArray(log_ent_coef=[entropy_coefficient.coef |> log])
end
function init_entropy_coefficient(entropy_coefficient::AutoEntropyCoefficient)
    ComponentArray(log_ent_coef=[entropy_coefficient.initial_value |> log])
end

function predict_actions(agent::SACAgent, observations::AbstractVector; deterministic::Bool=false, rng::AbstractRNG=agent.rng)
    policy = agent.policy
    ps = agent.train_state.parameters
    st = agent.train_state.states
    batched_obs = batch(observations, observation_space(policy))
    actions, st = predict_actions(policy, batched_obs, ps, st; deterministic, rng)
    #TODO: handle update of st? make agent mutable and set new train_state with @set?
    actions = process_action.(actions, Ref(action_space(policy)))
    return actions
end

function collect_trajectories(agent::SACAgent, env::AbstractParallelEnv, n_steps::Int,
    progress_meter::Union{Progress,Nothing}=nothing;
    callbacks::Union{Vector{<:AbstractCallback},Nothing}=nothing,
    use_random_actions::Bool=false)

    trajectories = OffPolicyTrajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [OffPolicyTrajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        all_good = true
        if !isnothing(callbacks)
            for callback in callbacks
                callback_good = on_step(callback, agent, env, current_trajectories, new_obs)
                all_good = all_good && callback_good
            end
            if !all_good
                @warn "Collecting trajectories stopped due to callback failure"
                return trajectories
            end
        end
        observations = new_obs
        if use_random_actions
            actions = rand(act_space, size(observations))
        else
        actions = predict_actions(agent, observations)
        end
        processed_actions = process_action.(actions, Ref(act_space))
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    current_trajectories[j].truncated_observation = last_observation
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    next_obs = new_obs[j]
                    current_trajectories[j].truncated_observation = next_obs
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = OffPolicyTrajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step=number_of_envs(env))
    end
    return trajectories
end

function collect_rollout!(buffer::ReplayBuffer, agent::SACAgent, env::AbstractParallelEnv,
    n_steps::Int, progress_meter::Union{Progress,Nothing}=nothing; kwargs...
)
    t_start = time()
    trajectories = collect_trajectories(agent, env, n_steps, progress_meter; kwargs...)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / t_collect

    for traj in trajectories
        push!(buffer, traj)
    end
    return fps
end

# Clean logging structure for SAC
struct SACTrainingStats{T<:AbstractFloat}
    actor_losses::Vector{T}
    critic_losses::Vector{T}
    entropy_losses::Vector{T}
    entropy_coefficients::Vector{T}
    q_values::Vector{T}
    learning_rates::Vector{T}
    grad_norms::Vector{T}
    fps::Vector{T}
    steps_taken::Vector{Int}
end

function SACTrainingStats{T}() where T<:AbstractFloat
    return SACTrainingStats{T}(T[], T[], T[], T[], T[], T[], T[], T[], T[])
end

function log_sac_training(agent::SACAgent, stats::SACTrainingStats, step::Int, env::AbstractParallelEnv)
    if !isnothing(agent.logger)
        set_step!(agent.logger, step)
        if !isempty(stats.actor_losses)
            log_value(agent.logger, "train/actor_loss", stats.actor_losses[end])
        end
        if !isempty(stats.critic_losses)
            log_value(agent.logger, "train/critic_loss", stats.critic_losses[end])
        end
        if !isempty(stats.entropy_losses)
            log_value(agent.logger, "train/entropy_loss", stats.entropy_losses[end])
        end
        if !isempty(stats.entropy_coefficients)
            log_value(agent.logger, "train/entropy_coefficient", stats.entropy_coefficients[end])
        end
        if !isempty(stats.q_values)
            log_value(agent.logger, "train/q_values", stats.q_values[end])
        end
        if !isempty(stats.learning_rates)
            log_value(agent.logger, "train/learning_rate", stats.learning_rates[end])
        end
        if !isempty(stats.grad_norms)
            log_value(agent.logger, "train/grad_norm", stats.grad_norms[end])
        end
        if !isempty(stats.fps)
            log_value(agent.logger, "env/fps", stats.fps[end])
        end
        log_value(agent.logger, "train/total_steps", steps_taken(agent))

        # Log episode statistics
        log_stats(env, agent.logger)
    end
end

function learn!(
    agent::SACAgent,
    env::AbstractParallelEnv,
    alg::OffPolicyAlgorithm,
    max_steps::Int; kwargs...
)
    replay_buffer = ReplayBuffer(observation_space(env), action_space(env), alg.buffer_capacity)
    learn!(agent, replay_buffer, env, alg, max_steps; kwargs...)
end

function learn!(
    agent::SACAgent,
    replay_buffer::ReplayBuffer,
    env::AbstractParallelEnv,
    alg::OffPolicyAlgorithm,
    max_steps::Int;
    #TODO: make callbacks an agent property?
    callbacks::Union{Vector{<:AbstractCallback},Nothing}=nothing,
    ad_type::Lux.Training.AbstractADType=AutoZygote()
)
    n_envs = number_of_envs(env)
    policy = agent.policy
    train_state = agent.train_state

    # Initialize training statistics
    T = eltype(alg.learning_rate)
    training_stats = SACTrainingStats{T}()

    # Calculate target entropy for automatic entropy coefficient
    target_entropy = get_target_entropy(alg.ent_coef, action_space(policy))

    # Check if we should update entropy coefficient
    update_entropy_coef = alg.ent_coef isa AutoEntropyCoefficient

    gradient_updates_performed = 0

    start_steps = alg.start_steps > 0 ? alg.start_steps : alg.train_freq
    n_steps = div(start_steps, number_of_envs(env))


    # Main training loop
    training_iteration = 0
    adjusted_train_freq = max(1, div(alg.train_freq, number_of_envs(env))) * number_of_envs(env)
    iterations = div(max_steps - n_steps * n_envs, adjusted_train_freq) + 1

    total_steps = n_steps * n_envs + adjusted_train_freq * (iterations - 1)

    #XXX: progress bar is not matching actual progress, its too quick to 99%
    progress_meter = Progress(total_steps, desc="Training...",
        showspeed=true, enabled=agent.verbose > 0
    )

    agent.verbose > 0 && @info "Starting SAC training with buffer size: $(length(replay_buffer)),
    start_steps: $(alg.start_steps), train_freq: $(alg.train_freq), number_of_envs: $(n_envs),
    adjusted_train_freq: $(adjusted_train_freq), iterations: $(iterations), total_steps: $(total_steps)"
    for training_iteration in 1:iterations  # Adjust this termination condition as needed
        # @info "Training iteration $training_iteration, collecting rollout ($n_steps steps)"
        # Collect experience
        fps = collect_rollout!(replay_buffer, agent, env, n_steps, progress_meter; callbacks,
            use_random_actions=training_iteration == 1 && alg.start_steps > 0)
        #set steps to train_freq after first (potentially larger) rollout
        push!(training_stats.fps, fps)
        add_step!(agent, n_steps * n_envs)
        n_steps = div(adjusted_train_freq, n_envs)

        # Perform gradient updates
        n_updates = get_gradient_steps(alg, adjusted_train_freq)
        data_loader = get_data_loader(replay_buffer, alg.batch_size, n_updates, true, true, agent.rng)

        for (i, batch_data) in enumerate(data_loader)
            # @info "Training iteration $training_iteration, batch $i, batch_size: $(size(batch_data.observations))"
            # Update entropy coefficient if using automatic entropy tuning
            if update_entropy_coef
                ent_train_state = agent.ent_train_state
                ent_data = (
                    observations=batch_data.observations,
                    policy_ps=train_state.parameters,
                    policy_st=train_state.states,
                    target_entropy=target_entropy,
                    target_ps=agent.Q_target_parameters,
                    target_st=agent.Q_target_states
                )
                ent_grad, ent_loss, _, ent_train_state = Lux.Training.compute_gradients(ad_type,
                    (model, ps, st, data) -> sac_ent_coef_loss(alg, policy, ps, st, data; rng=agent.rng),
                    ent_data,
                    ent_train_state
                )

                Lux.Training.apply_gradients!(ent_train_state, ent_grad)
                push!(training_stats.entropy_losses, ent_loss)
                @reset agent.ent_train_state = ent_train_state
            end


            train_state = agent.train_state
            # Update critic networks
            critic_data = (
                observations=batch_data.observations,
                actions=batch_data.actions,
                rewards=batch_data.rewards,
                terminated=batch_data.terminated,
                truncated=batch_data.truncated,
                next_observations=batch_data.next_observations,
                log_ent_coef=agent.ent_train_state.parameters,
                target_ps=agent.Q_target_parameters,
                target_st=agent.Q_target_states
            )

            critic_grad, critic_loss, critic_stats, train_state = Lux.Training.compute_gradients(ad_type,
                (model, ps, st, data) -> sac_critic_loss(alg, policy, ps, st, data; rng=agent.rng),
                critic_data,
                train_state
            )
            train_state = Lux.Training.apply_gradients(train_state, critic_grad)
            push!(training_stats.critic_losses, critic_loss)

            # Record Q-value statistics
            push!(training_stats.q_values, critic_stats["mean_q_values"])

            # Update actor network  
            actor_data = (
                observations=batch_data.observations,
                actions=batch_data.actions,
                rewards=batch_data.rewards,
                terminated=batch_data.terminated,
                truncated=batch_data.truncated,
                next_observations=batch_data.next_observations,
                log_ent_coef=agent.ent_train_state.parameters
            )

            actor_loss_grad, actor_loss, _, train_state = Lux.Training.compute_gradients(ad_type,
                (model, ps, st, data) -> sac_actor_loss(alg, policy, ps, st, data; rng=agent.rng),
                actor_data,
                train_state
            )
            zero_critic_grads!(actor_loss_grad, policy)
            @assert norm(actor_loss_grad.critic_head) < 1e-10 "Critic head gradient is not zero"
            train_state = Lux.Training.apply_gradients(train_state, actor_loss_grad)
            push!(training_stats.actor_losses, actor_loss)


            # Update target networks
            if gradient_updates_performed % alg.target_update_interval == 0
                @set agent.Q_target_states = copy_critic_states(policy, train_state.states)
                polyak_update!(agent.Q_target_parameters, train_state.parameters, alg.tau)

            end

            # Record statistics
            current_ent_coef = exp(agent.ent_train_state.parameters[1])
            push!(training_stats.entropy_coefficients, current_ent_coef)
            push!(training_stats.learning_rates, alg.learning_rate)

            total_grad_norm = sqrt(sum(norm(g)^2 for g in critic_grad) + sum(norm(g)^2 for g in actor_loss_grad))
            push!(training_stats.grad_norms, total_grad_norm)

            gradient_updates_performed += 1
            add_gradient_update!(agent)
        end

        # Log training statistics
        log_sac_training(agent, training_stats, steps_taken(agent), env)
    end

    # Return training statistics
    return agent, replay_buffer, training_stats
end

struct RolloutBuffer #spaces as parameters?
    observations::AbstractArray
    actions::AbstractArray
    rewards::AbstractArray
    advantages::AbstractArray
    returns::AbstractArray
    logprobs::AbstractArray
    values::AbstractArray
    gae_lambda::AbstractFloat
    gamma::AbstractFloat
    n_steps::Int
    n_envs::Int
end

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs


function RolloutBuffer(observation_space::AbstractBox, action_space::AbstractBox, gae_lambda::AbstractFloat, gamma::AbstractFloat, n_steps::Int, n_envs::Int)
    @assert observation_space.type == action_space.type "Observation and action spaces must have the same type"
    type = observation_space.type
    total_steps = n_steps * n_envs
    observations = Array{type,length(observation_space.shape) + 1}(undef, observation_space.shape..., total_steps)
    actions = Array{type,length(action_space.shape) + 1}(undef, action_space.shape..., total_steps)
    rewards = Array{type,1}(undef, total_steps)
    advantages = Array{type,1}(undef, total_steps)
    returns = Array{type,1}(undef, total_steps)
    logprobs = Array{type,1}(undef, total_steps)
    values = Array{type,1}(undef, total_steps)
    return RolloutBuffer(observations, actions, rewards, advantages, returns, logprobs, values, gae_lambda, gamma, n_steps, n_envs)
end

function reset!(rollout_buffer::RolloutBuffer)
    rollout_buffer.observations .= 0
    rollout_buffer.actions .= 0
    rollout_buffer.rewards .= 0
    rollout_buffer.advantages .= 0
    rollout_buffer.returns .= 0
    rollout_buffer.logprobs .= 0
    rollout_buffer.values .= 0
    nothing
end

mutable struct Trajectory{T<:AbstractFloat}
    observations::Vector{AbstractArray{T}}
    actions::Vector{AbstractArray{T}}
    rewards::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    terminated::Bool
    truncated::Bool
    bootstrap_value::Union{Nothing,T}  # Value of the next state for truncated episodes
    function Trajectory(observation_space::UniformBox, action_space::UniformBox)
        T = observation_space.type
        @assert T == action_space.type "Observation and action spaces must have the same type"
        observations = Vector{Array{T,length(observation_space.shape)}}[]
        actions = Vector{Array{T,length(action_space.shape)}}[]
        rewards = Vector{T}[]
        logprobs = Vector{T}[]
        values = Vector{T}[]
        terminated = false
        truncated = false
        bootstrap_value = nothing
        return new{T}(observations, actions, rewards, logprobs, values, terminated, truncated, bootstrap_value)
    end
end

Base.length(trajectory::Trajectory) = length(trajectory.rewards)
total_reward(trajectory::Trajectory) = sum(trajectory.rewards)


function collect_trajectories(agent::ActorCriticAgent, env::AbstractParallellEnv, n_steps::Int, progress_meter::Union{Progress,Nothing}=nothing)
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = env.n_envs
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        observations = new_obs
        actions, values, logprobs = get_action_and_values(agent, observations)
        @info "actions: $actions, values: $values, logprobs: $logprobs"
        processed_actions = process_action(actions, action_space(env))
        rewards, terminateds, truncateds, infos = step!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, eachslice(observations, dims=length(obs_space.shape) + 1)[j])
            push!(current_trajectories[j].actions, eachslice(actions, dims=length(act_space.shape) + 1)[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])

            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    terminal_value = predict_values(agent, last_observation)[1]
                    current_trajectories[j].bootstrap_value = terminal_value
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    next_obs = selectdim(new_obs, ndims(new_obs), j)
                    next_value = predict_values(agent, next_obs)[1]
                    current_trajectories[j].bootstrap_value = next_value
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step=env.n_envs)
    end
    return trajectories
end

function collect_rollouts!(rollout_buffer::RolloutBuffer, agent::ActorCriticAgent, env::AbstractEnv, progress_meter::Union{Progress,Nothing}=nothing)
    obs_space = observation_space(env)
    act_space = action_space(env)

    reset!(rollout_buffer)

    t_start = time()
    trajectories = collect_trajectories(agent, env, rollout_buffer.n_steps, progress_meter)
    t_collect = time() - t_start
    fps = sum(length.(trajectories)) / t_collect
    avg_ep_rew = mean(total_reward.(trajectories))

    if !isnothing(agent.logger)
        log_value(agent.logger, "env/avg_ep_rew", avg_ep_rew)
    end

    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer 
        traj_inds = positions[i]:positions[i+1]-1
        selectdim(rollout_buffer.observations, length(obs_space.shape) + 1, traj_inds) .= stack(traj.observations)
        selectdim(rollout_buffer.actions, length(act_space.shape) + 1, traj_inds) .= stack(traj.actions)
        rollout_buffer.rewards[traj_inds] .= traj.rewards
        rollout_buffer.logprobs[traj_inds] .= traj.logprobs
        rollout_buffer.values[traj_inds] .= traj.values
        #compute advantages and returns
        compute_advantages!(@view(rollout_buffer.advantages[traj_inds]),
            traj, rollout_buffer.gamma, rollout_buffer.gae_lambda)
        rollout_buffer.returns[traj_inds] .= rollout_buffer.advantages[traj_inds] .+ rollout_buffer.values[traj_inds]
    end
    return fps
end

function compute_advantages!(advantages::AbstractArray, traj::Trajectory, gamma::AbstractFloat, gae_lambda::AbstractFloat)
    n = length(traj.rewards)

    # For terminated episodes, no bootstrapping (bootstrap_value should be nothing)
    # For truncated episodes or rollout-limited trajectories, bootstrap with the next state value
    if traj.terminated || isnothing(traj.bootstrap_value)
        # No bootstrapping for terminated episodes
        delta = traj.rewards[end] - traj.values[end]
    else
        # Bootstrap for truncated or rollout-limited trajectories
        delta = traj.rewards[end] + gamma * traj.bootstrap_value - traj.values[end]
    end

    advantages[end] = delta

    # Compute advantages for earlier steps using the standard GAE recursion
    for i in (n-1):-1:1
        delta = traj.rewards[i] + gamma * traj.values[i+1] - traj.values[i]
        advantages[i] = delta + gamma * gae_lambda * advantages[i+1]
    end

    nothing
end
struct RolloutBuffer{T<:AbstractFloat,O,A}
    observations::Array{O}
    actions::Array{A}
    rewards::Vector{T}
    advantages::Vector{T}
    returns::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    gae_lambda::T
    gamma::T
    n_steps::Int
    n_envs::Int
end

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs

#TODO:fix types here
function RolloutBuffer(observation_space::AbstractSpace, action_space::AbstractSpace, gae_lambda::T, gamma::T, n_steps::Int, n_envs::Int) where {T<:AbstractFloat}
    total_steps = n_steps * n_envs
    obs_type = eltype(observation_space)
    action_type = eltype(action_space)
    observations = Array{obs_type}(undef, size(observation_space)..., total_steps)
    actions = Array{action_type}(undef, size(action_space)..., total_steps)
    rewards = Vector{T}(undef, total_steps)
    advantages = Vector{T}(undef, total_steps)
    returns = Vector{T}(undef, total_steps)
    logprobs = Vector{T}(undef, total_steps)
    values = Vector{T}(undef, total_steps)
    return RolloutBuffer{T,obs_type,action_type}(observations, actions, rewards, advantages, returns, logprobs, values, gae_lambda, gamma, n_steps, n_envs)
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

mutable struct Trajectory{T<:AbstractFloat,O,A}
    observations::Vector{Array{O}}
    actions::Vector{Array{A}}
    rewards::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    terminated::Bool
    truncated::Bool
    bootstrap_value::Union{Nothing,T}  # Value of the next state for truncated episodes
end
function Trajectory{T}(observation_space::AbstractSpace, action_space::AbstractSpace) where {T<:AbstractFloat}
    obs_type = eltype(observation_space)
    action_type = eltype(action_space)
    observations = Array{obs_type}[]
    actions = Array{action_type}[]
    rewards = T[]
    logprobs = T[]
    values = T[]
    terminated = false
    truncated = false
    bootstrap_value = nothing
    return Trajectory{T,obs_type,action_type}(observations, actions, rewards, logprobs, values, terminated, truncated, bootstrap_value)
end

Trajectory(observation_space::AbstractSpace, action_space::AbstractSpace) = Trajectory{Float32}(observation_space, action_space)

Base.length(trajectory::Trajectory) = length(trajectory.rewards)
total_reward(trajectory::Trajectory) = sum(trajectory.rewards)


function collect_trajectories(agent::ActorCriticAgent, env::AbstractParallellEnv, n_steps::Int,
    progress_meter::Union{Progress,Nothing}=nothing)
    # reset!(env)
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        observations = new_obs
        actions, values, logprobs = get_action_and_values(agent, observations)
        processed_actions = process_action.(actions, Ref(act_space))
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    terminal_value = predict_values(agent, [last_observation])[1]
                    current_trajectories[j].bootstrap_value = terminal_value
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    next_obs = new_obs[j]
                    next_value = predict_values(agent, [next_obs])[1]
                    current_trajectories[j].bootstrap_value = next_value
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step=number_of_envs(env))
    end
    return trajectories
end

function collect_rollouts!(rollout_buffer::RolloutBuffer, agent::ActorCriticAgent, env::AbstractEnv, progress_meter::Union{Progress,Nothing}=nothing)
    # reset!(env) #we dont reset the, we continue from where we left off
    obs_space = observation_space(env)
    act_space = action_space(env)

    reset!(rollout_buffer)

    t_start = time()
    trajectories = collect_trajectories(agent, env, rollout_buffer.n_steps, progress_meter)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / t_collect

    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer 
        traj_inds = positions[i]:positions[i+1]-1
        selectdim(rollout_buffer.observations, length(size(obs_space)) + 1, traj_inds) .= stack(traj.observations)
        selectdim(rollout_buffer.actions, length(size(act_space)) + 1, traj_inds) .= stack(traj.actions)
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
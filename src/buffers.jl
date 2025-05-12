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
    observations = Array{type, length(observation_space.shape)+1}(undef, observation_space.shape..., total_steps)
    actions = Array{type, length(action_space.shape)+1}(undef, action_space.shape..., total_steps)
    rewards = Array{type, 1}(undef, total_steps)
    advantages = Array{type, 1}(undef, total_steps)
    returns = Array{type, 1}(undef, total_steps)
    logprobs = Array{type, 1}(undef, total_steps)
    values = Array{type, 1}(undef, total_steps)
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

mutable struct Trajectory
    observations::Vector{AbstractArray}
    actions::Vector{AbstractArray}
    rewards::Vector{Number}
    logprobs::Vector{Number}
    values::Vector{Number}
    terminated::Bool
    truncated::Bool
    function Trajectory(observation_space::UniformBox, action_space::UniformBox)
        observations = Vector{Array{observation_space.type, length(observation_space.shape)}}[]
        actions = Vector{Array{action_space.type, length(action_space.shape)}}[]
        rewards = Vector{action_space.type}[]
        logprobs = Vector{action_space.type}[]
        values = Vector{action_space.type}[]
        terminated = false
        truncated = false
        return new(observations, actions, rewards, logprobs, values, terminated, truncated)
    end
end

Base.length(trajectory::Trajectory) = length(trajectory.rewards)



function collect_trajectories(agent::ActorCriticAgent, env::AbstractParallellEnv, n_steps::Int, gamma::AbstractFloat, progress_meter::Union{Progress, Nothing}=nothing)
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = env.n_envs
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    for i in 1:n_steps
        observations = observe(env)
        actions, values, logprobs = get_action_and_values(agent, observations)
        rewards, terminateds, truncateds, infos = step!(env, actions)
        @show terminateds
        @show truncateds
        for j in 1:n_envs
            push!(current_trajectories[j].observations, eachslice(observations, dims=length(obs_space.shape)+1)[j])
            push!(current_trajectories[j].actions, eachslice(actions, dims=length(act_space.shape)+1)[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])
            
            if terminateds[j] || truncateds[j] || i == n_steps
               current_trajectories[j].terminated = terminateds[j]
               current_trajectories[j].truncated = truncateds[j]

                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    #ignore derivatives here? maybe not necessary
                    terminal_value = predict_values(agent, last_observation)[1]
                    current_trajectories[j].rewards[end] += terminal_value*gamma
                end
                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step=env.n_envs)
    end
    return trajectories
end

function collect_rollouts!(rollout_buffer::RolloutBuffer, agent::ActorCriticAgent, env::AbstractEnv, progress_meter::Union{Progress, Nothing}=nothing)
    obs_space = observation_space(env)
    act_space = action_space(env)

    reset!(rollout_buffer)

    t_start = time()
    trajectories = collect_trajectories(agent, env, rollout_buffer.n_steps, rollout_buffer.gamma, progress_meter)
    t_collect = time() - t_start
    fps = sum(length.(trajectories)) / t_collect

    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer 
        traj_inds = positions[i]:positions[i+1]-1
        selectdim(rollout_buffer.observations, length(obs_space.shape)+1, traj_inds) .= stack(traj.observations)
        selectdim(rollout_buffer.actions, length(act_space.shape)+1, traj_inds) .= stack(traj.actions)
        rollout_buffer.rewards[traj_inds] .= traj.rewards
        rollout_buffer.logprobs[traj_inds] .= traj.logprobs
        rollout_buffer.values[traj_inds] .= traj.values
        #compute advantages and returns
        compute_advantages!(rollout_buffer.advantages[traj_inds],
                            traj, rollout_buffer.gamma, rollout_buffer.gae_lambda)
        rollout_buffer.returns[traj_inds] .= rollout_buffer.advantages[traj_inds] .+ rollout_buffer.values[traj_inds]
    end
    return fps
end

function compute_advantages!(advantages::AbstractArray, traj::Trajectory, gamma::AbstractFloat, gae_lambda::AbstractFloat)
    delta = traj.rewards[end] - traj.values[end]
    advantages[end] = delta
    for i in length(traj.rewards)-1:-1:1
        delta = traj.rewards[i] + gamma * traj.values[i+1] - traj.values[i]
        advantages[i] = delta + gamma * gae_lambda * advantages[i+1]
    end
    nothing
end
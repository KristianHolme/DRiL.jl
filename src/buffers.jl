mutable struct Trajectory
    observations::Vector{AbstractArray}
    actions::Vector{AbstractArray}
    rewards::Vector{Number}
    logprobs::Vector{Number}
    values::Vector{Number}
    terminated::Bool
    truncated::Bool
end

function Trajectory(observation_space::UniformBox, action_space::UniformBox)
    observations = Vector{Array{observation_space.type, length(observation_space.shape)}}[]
    actions = Vector{Array{action_space.type, length(action_space.shape)}}[]
    rewards = Vector{action_space.type}[]
    logprobs = Vector{action_space.type}[]
    values = Vector{action_space.type}[]
    terminated = false
    truncated = false
    return Trajectory(observations, actions, rewards, logprobs, values, terminated, truncated)
end

function collect_trajectories(agent::AbstractAgent, env::AbstractParallellEnv, n_steps::Int)
    trajectories = Trajectory[]
    observation_space = env.observation_space
    action_space = env.action_space
    current_trajectories = [Trajectory(observation_space, action_space) for _ in 1:n_envs]
    for i in 1:n_steps
        observations = observe(env)
        actions, logprobs, _, values = get_action_and_value(agent, observations)
        rewards, terminateds, truncateds, infos = step!(env, actions)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])
            if terminateds[j] || truncateds[j]
               current_trajectories[j].terminated = terminateds[j]
               current_trajectories[j].truncated = truncateds[j]

                if truncateds[j] haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    #ignore derivatives here? maybe not necessary
                    terminal_value = predict_values(agent, last_observation)
                    current_trajectories[j].rewards[end] += terminal_value
                end
                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(observation_space, action_space)
            end
        end
    end
    return trajectories
end

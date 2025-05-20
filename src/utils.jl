function collect_trajectory(agent::ActorCriticAgent, env::AbstractEnv; max_steps::Union{Int,Nothing}=nothing)
    reset!(env)
    observations = []
    actions = []
    rewards = []
    while !(terminated(env) || truncated(env))
        observation = observe(env)
        push!(observations, observation)
        observation = reshape(observation, size(observation)..., 1)
        action = predict_actions(agent, observation, deterministic=true)
        action = selectdim(action, ndims(action), 1)
        push!(actions, action)
        reward = act!(env, action)
        push!(rewards, reward)
        if max_steps !== nothing && length(observations) >= max_steps
            break
        end
    end
    return observations, actions, rewards
end
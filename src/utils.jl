function collect_trajectory(agent::ActorCriticAgent, env::AbstractEnv;
    max_steps::Union{Int,Nothing}=nothing, norm_env::Union{NormalizeWrapperEnv,Nothing}=nothing)
    reset!(env)
    original_training = is_training(env)
    set_training!(env, false)
    observations = []
    actions = []
    rewards = []
    while !(terminated(env) || truncated(env))
        observation = observe(env)
        if env isa ScalingWrapperEnv
            original_observation = unscale_observation(env, observation)
        else
            original_observation = observation
        end
        push!(observations, original_observation)
        observation = reshape(observation, size(observation)..., 1)
        if norm_env !== nothing
            observation = normalize_obs(norm_env, observation)
        end

        agent_action = predict_actions(agent, observation, deterministic=true)
        agent_action = selectdim(agent_action, ndims(agent_action), 1)
        if env isa ScalingWrapperEnv
            env_action = unscale_action(env, agent_action)
        else
            env_action = agent_action
        end
        push!(actions, env_action)
        reward = act!(env, agent_action)

        push!(rewards, reward)
        if max_steps !== nothing && length(observations) >= max_steps
            @warn "Max steps reached"
            break
        end
    end
    set_training!(env, original_training)
    return observations, actions, rewards
end
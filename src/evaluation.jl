function is_monitored(env::AbstractParallelEnv)
    monitored = false
    while env isa AbstractParallelEnvWrapper
        monitored = env isa MonitorWrapperEnv
        env = unwrap(env)
    end
    return monitored
end

#TODO:update docstring
"""
    evaluate_agent(agent, env; kwargs...)

Evaluate a policy/agent for a specified number of episodes and return performance statistics.

# Arguments
- `agent`: The agent to evaluate (must implement `predict` method)
- `env`: The environment to evaluate on (single env or parallel env)

# Keyword Arguments
- `n_eval_episodes::Int = 10`: Number of episodes to evaluate
- `deterministic::Bool = true`: Whether to use deterministic actions
- `render::Bool = false`: Whether to render the environment
- `callback::Union{Nothing, Function} = nothing`: Optional callback function called after each step
- `reward_threshold::Union{Nothing, Real} = nothing`: Minimum expected mean reward (throws error if not met)
- `return_episode_rewards::Bool = false`: If true, returns individual episode rewards and lengths
- `warn::Bool = true`: Whether to warn about missing Monitor wrapper
- `rng::AbstractRNG = Random.default_rng()`: Random number generator for reproducible evaluation

# Returns
- If `return_episode_rewards = false`: `(mean_reward::Float64, std_reward::Float64)`
- If `return_episode_rewards = true`: `(episode_rewards::Vector{Float64}, episode_lengths::Vector{Int})`

# Notes
- Episodes are distributed evenly across parallel environments to remove bias
- If environment is wrapped with Monitor, episode statistics from Monitor are used
- Otherwise, rewards and lengths are tracked manually during evaluation
- For environments with reward/length modifying wrappers, consider using Monitor wrapper

# Examples
```julia
# Basic evaluation
mean_reward, std_reward = evaluate_agent(agent, env; n_eval_episodes=20)

# Get individual episode data
episode_rewards, episode_lengths = evaluate_agent(agent, env; 
    return_episode_rewards=true, deterministic=false)

# Evaluation with threshold check
mean_reward, std_reward = evaluate_agent(agent, env; 
    reward_threshold=100.0, n_eval_episodes=50)
```
"""
function evaluate_agent(
    agent,
    env::AbstractParallelEnv;
    n_eval_episodes::Int=10,
    deterministic::Bool=true,
    reward_threshold::Union{Nothing,Real}=nothing,
    return_stats::Bool=true,
    warn::Bool=true,
    rng::AbstractRNG=agent.rng,
    show_progress::Bool=false
)

    # Check if environment is wrapped with Monitor (when Monitor is implemented)
    is_monitor_wrapped = is_monitored(env)

    if !is_monitor_wrapped && warn
        @warn """Evaluation environment is not wrapped with a Monitor wrapper. 
                 This may result in reporting modified episode lengths and rewards, 
                 if other wrappers happen to modify these. Consider wrapping 
                 environment first with Monitor wrapper."""
    end

    # Initialize tracking variables
    T = eltype(observation_space(env))
    episode_rewards = T[]
    episode_lengths = Int[]

    n_envs = number_of_envs(env)
    # For parallel environments, distribute episodes evenly

    current_rewards = zeros(T, n_envs)
    current_lengths = zeros(Int, n_envs)

    # Reset environment
    reset!(env)
    observations = observe(env)

    p = Progress(n_eval_episodes; enabled=show_progress)

    while length(episode_rewards) < n_eval_episodes
        # Get actions from agent
        actions = predict_actions(agent, observations; deterministic, rng)

        # Take step in environment
        step_rewards = act!(env, actions)
        terminateds = terminated(env)
        truncateds = truncated(env)
        infos = get_info(env)
        current_rewards .+= step_rewards
        current_lengths .+= 1
        observations = observe(env)

        # Process each environment
        dones = terminateds .| truncateds
        for i in 1:n_envs
            if length(episode_rewards) < n_eval_episodes
                # Check if episode ended
                if dones[i]
                    next!(p)
                    if is_monitor_wrapped && haskey(infos[i], "episode")
                        # Use Monitor statistics if available
                        push!(episode_rewards, infos[i]["episode"]["r"])
                        push!(episode_lengths, infos[i]["episode"]["l"])
                    else
                        # Use manually tracked statistics
                        push!(episode_rewards, current_rewards[i])
                        push!(episode_lengths, current_lengths[i])
                    end

                    current_rewards[i] = 0
                    current_lengths[i] = 0
                end
            end
        end
    end

    # Calculate statistics
    mean_reward = mean(episode_rewards)
    std_reward = std(episode_rewards)

    # Check reward threshold if specified
    if reward_threshold !== nothing
        if mean_reward < reward_threshold
            error("Mean reward below threshold: $(round(mean_reward, digits=2)) < $(reward_threshold)")
        end
    end

    # Return results
    if return_stats
        return (; mean_reward, std_reward, mean_length=mean(episode_lengths), std_length=std(episode_lengths))
    else
        return episode_rewards, episode_lengths
    end
end
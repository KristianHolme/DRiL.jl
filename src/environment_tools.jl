function unwrap_all(env::AbstractEnv)
    is_wrapper = true
    while is_wrapper
        env = unwrap(env)
        is_wrapper = is_wrapper(env)
    end
    return env
end

function observation_space(env::AbstractParallellEnv)
    return observation_space(env.envs[1])
end

function action_space(env::AbstractParallellEnv)
    return action_space(env.envs[1])
end

struct MultiThreadedParallelEnv{E<:AbstractEnv} <: AbstractParallellEnv
    envs::Vector{E}
    function MultiThreadedParallelEnv(envs::Vector{E}) where E<:AbstractEnv
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    @threads for env in env.envs
        reset!(env)
    end
    nothing
end

function observe(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    obs_space = observation_space(env)
    type = eltype(obs_space)
    observations = Array{type,length(obs_space.shape) + 1}(undef, (obs_space.shape..., length(env.envs)))
    @threads for i in 1:length(env.envs)
        selectdim(observations, length(obs_space.shape) + 1, i) .= observe(env.envs[i])
    end
    return observations
end

function terminated(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    return terminated.(env.envs)
end

function truncated(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    return truncated.(env.envs)
end

function action_space(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    return action_space(env.envs[1])
end

function observation_space(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    return observation_space(env.envs[1])
end

function get_info(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    return get_info.(env.envs)
end

function step!(env::MultiThreadedParallelEnv{E}, action) where E<:AbstractEnv
    # @show action
    type = eltype(observation_space(env))
    rewards = Vector{type}(undef, length(env.envs))
    terminateds = Vector{Bool}(undef, length(env.envs))
    truncateds = Vector{Bool}(undef, length(env.envs))
    infos = Vector{Dict{String,Any}}(undef, length(env.envs))
    action_dims = length(action_space(env).shape)
    @assert size(action) == (action_space(env).shape..., length(env.envs)) "Action must be of shape $((action_space(env).shape..., length(env.envs)))"
    @threads for i in 1:length(env.envs)
        # @info "Stepping env $i"
        local_action = selectdim(action, action_dims + 1, i)
        # @show local_action i
        rewards[i] = act!(env.envs[i], local_action)
        # @info "Reward: $rewards[i]"
        terminateds[i] = terminated(env.envs[i])
        truncateds[i] = truncated(env.envs[i])
        infos[i] = get_info(env.envs[i])
        if truncateds[i]
            infos[i]["terminal_observation"] = observe(env.envs[i])
        end
        if terminateds[i] || truncateds[i]
            reset!(env.envs[i])
        end
    end
    return rewards, terminateds, truncateds, infos
end

number_of_envs(env::MultiThreadedParallelEnv) = length(env.envs)

struct ScalingWrapperEnv{E<:AbstractEnv,O<:AbstractSpace,A<:AbstractSpace} <: AbstractEnvWrapper{E}
    env::E
    observation_space::O
    action_space::A
    orig_observation_space::O
    orig_action_space::A
end

function ScalingWrapperEnv(env::E) where {E<:AbstractEnv}
    orig_obs_space = observation_space(env)
    orig_act_space = action_space(env)

    return ScalingWrapperEnv(env, orig_obs_space, orig_act_space)
end

# Specific constructor for UniformBox spaces
function ScalingWrapperEnv(env::E,
    original_obs_space::UniformBox,
    original_act_space::UniformBox
) where {E<:AbstractEnv}
    # Create new observation space with bounds [-1, 1]
    T_obs = eltype(original_obs_space)
    T_act = eltype(original_act_space)
    scaled_obs_space = @set original_obs_space.low = T_obs(-1)
    scaled_obs_space = @set scaled_obs_space.high = T_obs(1)

    # Create new action space with bounds [-1, 1]
    scaled_act_space = @set original_act_space.low = T_act(-1)
    scaled_act_space = @set scaled_act_space.high = T_act(1)

    return ScalingWrapperEnv{E,UniformBox,UniformBox}(env, scaled_obs_space, scaled_act_space, original_obs_space, original_act_space)
end
DRiL.unwrap(env::ScalingWrapperEnv) = env.env

function observation_space(env::ScalingWrapperEnv)
    return env.observation_space
end

function action_space(env::ScalingWrapperEnv)
    return env.action_space
end

function reset!(env::ScalingWrapperEnv)
    reset!(env.env)
    nothing
end

function observe(env::ScalingWrapperEnv{E,UniformBox,UniformBox}) where E
    orig_obs = observe(env.env)
    orig_space = observation_space(env.env)

    # Scale observation from original space to [-1, 1]
    scaled_obs = 2 .* (orig_obs .- orig_space.low) ./ (orig_space.high .- orig_space.low) .- 1
    return scaled_obs
end

function act!(env::ScalingWrapperEnv{E,UniformBox,UniformBox}, action) where E
    orig_space = action_space(env.env)

    # Scale action from [-1, 1] to original space
    orig_action = (action .+ 1) ./ 2 .* (orig_space.high .- orig_space.low) .+ orig_space.low
    return act!(env.env, orig_action)
end

function terminated(env::ScalingWrapperEnv)
    return terminated(env.env)
end

function truncated(env::ScalingWrapperEnv)
    return truncated(env.env)
end

function get_info(env::ScalingWrapperEnv)
    return get_info(env.env)
end

# Random.seed! extensions for environments
"""
    Random.seed!(env::AbstractEnv, seed::Integer)

Seed an environment's internal RNG. Environments should have an `rng` field 
that gets seeded for reproducible behavior.
"""
function Random.seed!(env::AbstractEnv, seed::Integer)
    if hasfield(typeof(env), :rng)
        Random.seed!(env.rng, seed)
    else
        @warn "Environment $(typeof(env)) does not have an rng field - seeding has no effect"
    end
    return env
end

"""
    Random.seed!(env::AbstractParallellEnv, seed::Integer)

Seed all sub-environments in a parallel environment with incremented seeds.
Each sub-environment gets seeded with `seed + i - 1` where `i` is the environment index.
"""
function Random.seed!(env::AbstractParallellEnv, seed::Integer)
    for (i, sub_env) in enumerate(env.envs)
        Random.seed!(sub_env, seed + i - 1)
    end
    return env
end

"""
    Random.seed!(env::ScalingWrapperEnv, seed::Integer)

Seed a wrapped environment by forwarding the seed to the underlying environment.
"""
function Random.seed!(env::ScalingWrapperEnv, seed::Integer)
    Random.seed!(env.env, seed)
    return env
end

# Running mean and standard deviation tracker for normalization
"""
    RunningMeanStd{T}

Tracks running mean and standard deviation using Welford's online algorithm.
Similar to stable-baselines3's RunningMeanStd but optimized for Julia.
"""
mutable struct RunningMeanStd{T<:AbstractFloat}
    mean::Array{T}
    var::Array{T}
    count::Int

    function RunningMeanStd{T}(shape::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
        return new{T}(zeros(T, shape), ones(T, shape), 0)
    end
end

RunningMeanStd(shape::Tuple{Vararg{Int}}) = RunningMeanStd{Float32}(shape)
RunningMeanStd(::Type{T}, shape::Tuple{Vararg{Int}}) where {T<:AbstractFloat} = RunningMeanStd{T}(shape)

function update!(rms::RunningMeanStd{T}, batch::AbstractArray{T}) where {T}
    batch_mean = mean(batch, dims=ndims(batch))
    batch_var = var(batch, dims=ndims(batch), corrected=false)
    batch_count = size(batch, ndims(batch))
    update_from_moments!(rms, batch_mean, batch_var, batch_count)
end

function update_from_moments!(rms::RunningMeanStd{T}, batch_mean::AbstractArray{T},
    batch_var::AbstractArray{T}, batch_count::Int) where {T}
    if rms.count == 0
        rms.mean .= dropdims(batch_mean, dims=ndims(batch_mean))
        rms.var .= dropdims(batch_var, dims=ndims(batch_var))
        rms.count = batch_count
    else
        delta = dropdims(batch_mean, dims=ndims(batch_mean)) .- rms.mean
        total_count = rms.count + batch_count

        new_mean = rms.mean .+ delta .* batch_count ./ total_count
        m_a = rms.var .* rms.count
        m_b = dropdims(batch_var, dims=ndims(batch_var)) .* batch_count
        M2 = m_a .+ m_b .+ delta .^ 2 .* rms.count .* batch_count ./ total_count
        new_var = M2 ./ total_count

        rms.mean .= new_mean
        rms.var .= new_var
        rms.count = total_count
    end
end

struct NormalizeWrapperEnv{E<:AbstractParallellEnv,T<:AbstractFloat} <: AbstractParallellEnvWrapper{E}
    env::E
    obs_rms::RunningMeanStd{T}
    ret_rms::RunningMeanStd{T}
    returns::Vector{T}

    # Configuration
    training::Bool
    norm_obs::Bool
    norm_reward::Bool
    clip_obs::T
    clip_reward::T
    gamma::T
    epsilon::T

    # Cache for original observations/rewards
    old_obs::Array{T}
    old_rewards::Vector{T}
end
function NormalizeWrapperEnv{E,T}(
    env::E;
    training::Bool=true,
    norm_obs::Bool=true,
    norm_reward::Bool=true,
    clip_obs::T=T(10.0),
    clip_reward::T=T(10.0),
    gamma::T=T(0.99),
    epsilon::T=T(1e-8)
) where {E<:AbstractParallellEnv,T<:AbstractFloat}

    obs_space = observation_space(env)
    n_envs = number_of_envs(env)

    # Initialize running statistics
    obs_rms = RunningMeanStd(T, obs_space.shape)
    ret_rms = RunningMeanStd(T, ())
    returns = zeros(T, n_envs)

    # Initialize cache arrays
    old_obs = Array{T}(undef, obs_space.shape..., n_envs)
    old_rewards = Vector{T}(undef, n_envs)

    return NormalizeWrapperEnv{E,T}(env, obs_rms, ret_rms, returns, training, norm_obs, norm_reward,
        clip_obs, clip_reward, gamma, epsilon, old_obs, old_rewards)
end
DRiL.unwrap(env::NormalizeWrapperEnv) = env.env

# Convenience constructor
function NormalizeWrapperEnv(env::E; kwargs...) where {E<:AbstractParallellEnv}
    return NormalizeWrapperEnv{E,Float32}(env; kwargs...)
end

# Forward basic properties
observation_space(env::NormalizeWrapperEnv) = observation_space(env.env)
action_space(env::NormalizeWrapperEnv) = action_space(env.env)
number_of_envs(env::NormalizeWrapperEnv) = number_of_envs(env.env)

function reset!(env::NormalizeWrapperEnv{E,T}) where {E,T}
    reset!(env.env)
    obs = observe(env.env)
    env.old_obs .= obs
    env.returns .= zero(T)

    # Update observation statistics if in training mode
    if env.training && env.norm_obs
        update!(env.obs_rms, obs)
    end

    return normalize_obs(env, obs)
end

function observe(env::NormalizeWrapperEnv{E,T}) where {E,T}
    obs = observe(env.env)
    env.old_obs .= obs
    return normalize_obs(env, obs)
end

function step!(env::NormalizeWrapperEnv{E,T}, action) where {E,T}
    rewards, terminateds, truncateds, infos = step!(env.env, action)
    obs = observe(env.env)

    env.old_obs .= obs
    env.old_rewards .= rewards

    # Update observation statistics if in training mode
    if env.training && env.norm_obs
        update!(env.obs_rms, obs)
    end

    # Update reward statistics and normalize
    if env.training && env.norm_reward
        update_reward_stats!(env, rewards)
    end

    normalized_obs = normalize_obs(env, obs)
    normalized_rewards = normalize_reward(env, rewards)

    # Handle terminal observations in info
    for (i, (term, trunc)) in enumerate(zip(terminateds, truncateds))
        if term || trunc
            if haskey(infos[i], "terminal_observation")
                infos[i]["terminal_observation"] = normalize_obs(env, infos[i]["terminal_observation"])
            end
            env.returns[i] = zero(T)
        end
    end

    return normalized_rewards, terminateds, truncateds, infos
end

function update_reward_stats!(env::NormalizeWrapperEnv{E,T}, rewards::Vector{T}) where {E,T}
    env.returns .= env.returns .* env.gamma .+ rewards
    # Update return statistics (single value, so we reshape for consistency)
    update!(env.ret_rms, reshape(env.returns, 1, length(env.returns)))
end

function normalize_obs(env::NormalizeWrapperEnv{E,T}, obs::AbstractArray{T}) where {E,T}
    if !env.norm_obs
        return obs
    end

    # Normalize using running statistics
    normalized = (obs .- env.obs_rms.mean) ./ sqrt.(env.obs_rms.var .+ env.epsilon)
    return clamp.(normalized, -env.clip_obs, env.clip_obs)
end

function normalize_reward(env::NormalizeWrapperEnv{E,T}, rewards::Vector{T}) where {E,T}
    if !env.norm_reward
        return rewards
    end

    # Normalize rewards using return statistics
    normalized = rewards ./ sqrt(env.ret_rms.var[1] + env.epsilon)
    return clamp.(normalized, -env.clip_reward, env.clip_reward)
end

function unnormalize_obs(env::NormalizeWrapperEnv{E,T}, obs::AbstractArray{T}) where {E,T}
    if !env.norm_obs
        return obs
    end
    return obs .* sqrt.(env.obs_rms.var .+ env.epsilon) .+ env.obs_rms.mean
end

function unnormalize_reward(env::NormalizeWrapperEnv{E,T}, rewards::Vector{T}) where {E,T}
    if !env.norm_reward
        return rewards
    end
    return rewards .* sqrt(env.ret_rms.var[1] + env.epsilon)
end

# Get original (unnormalized) observations and rewards
get_original_obs(env::NormalizeWrapperEnv) = copy(env.old_obs)
get_original_rewards(env::NormalizeWrapperEnv) = copy(env.old_rewards)

# Forward other methods
terminated(env::NormalizeWrapperEnv) = terminated(env.env)
truncated(env::NormalizeWrapperEnv) = truncated(env.env)
get_info(env::NormalizeWrapperEnv) = get_info(env.env)

function Random.seed!(env::NormalizeWrapperEnv, seed::Integer)
    Random.seed!(env.env, seed)
    return env
end

# Training mode control
set_training!(env::NormalizeWrapperEnv, training::Bool) = @reset env.training = training
is_training(env::NormalizeWrapperEnv) = env.training

# Save/load functionality for normalization statistics
"""
    save_normalization_stats(env::NormalizeWrapperEnv, filepath::String)

Save the normalization statistics (running mean/std) to a file using JLD2.
"""
function save_normalization_stats(env::NormalizeWrapperEnv, filepath::String)
    save(filepath, Dict(
        "obs_mean" => env.obs_rms.mean,
        "obs_var" => env.obs_rms.var,
        "obs_count" => env.obs_rms.count,
        "ret_mean" => env.ret_rms.mean,
        "ret_var" => env.ret_rms.var,
        "ret_count" => env.ret_rms.count,
        "clip_obs" => env.clip_obs,
        "clip_reward" => env.clip_reward,
        "gamma" => env.gamma,
        "epsilon" => env.epsilon
    ))
end

"""
    load_normalization_stats!(env::NormalizeWrapperEnv, filepath::String)

Load normalization statistics from a file into the environment using JLD2.
"""
function load_normalization_stats!(env::NormalizeWrapperEnv{E,T}, filepath::String) where {E,T}
    stats = load(filepath)

    # Load observation statistics
    env.obs_rms.mean .= T.(stats["obs_mean"])
    env.obs_rms.var .= T.(stats["obs_var"])
    env.obs_rms.count = stats["obs_count"]

    # Load return statistics  
    env.ret_rms.mean .= T.(stats["ret_mean"])
    env.ret_rms.var .= T.(stats["ret_var"])
    env.ret_rms.count = stats["ret_count"]

    return env
end

#syncs the eval env stats to be same as training env
function sync_normalization_stats!(eval_env::NormalizeWrapperEnv, train_env::NormalizeWrapperEnv)
    eval_env.obs_rms.mean .= train_env.obs_rms.mean
    eval_env.obs_rms.var .= train_env.obs_rms.var
    eval_env.obs_rms.count = train_env.obs_rms.count
    eval_env.ret_rms.mean .= train_env.ret_rms.mean
    eval_env.ret_rms.var .= train_env.ret_rms.var
    eval_env.ret_rms.count = train_env.ret_rms.count
    eval_env.returns .= train_env.returns
    nothing
end


struct EpisodeStats{T<:AbstractFloat}
    episode_returns::CircularBuffer{T}
    episode_lengths::CircularBuffer{Int}
end
function EpisodeStats{T}(stats_window::Int) where T
    return EpisodeStats{T}(CircularBuffer{T}(stats_window), CircularBuffer{Int}(stats_window))
end

struct MonitorWrapperEnv{E<:AbstractParallellEnv,T} <: AbstractParallellEnvWrapper{E} where T<:AbstractFloat
    env::E
    current_episode_lengths::Vector{Int}
    current_episode_returns::Vector{T}
    episode_stats::EpisodeStats{T}
end

function MonitorWrapperEnv(env::E, stats_window::Int=100) where E<:AbstractParallellEnv
    T = eltype(observation_space(env))
    return MonitorWrapperEnv{E,T}(
        env,
        zeros(Int, number_of_envs(env)),
        zeros(T, number_of_envs(env)),
        EpisodeStats{T}(stats_window)
    )
end

#TODO clean up this so its not necessary to forward all the methods
observe(monitor_env::MonitorWrapperEnv) = observe(monitor_env.env)
terminated(monitor_env::MonitorWrapperEnv) = terminated(monitor_env.env)
truncated(monitor_env::MonitorWrapperEnv) = truncated(monitor_env.env)
get_info(monitor_env::MonitorWrapperEnv) = get_info(monitor_env.env)
action_space(monitor_env::MonitorWrapperEnv) = action_space(monitor_env.env)
observation_space(monitor_env::MonitorWrapperEnv) = observation_space(monitor_env.env)
number_of_envs(monitor_env::MonitorWrapperEnv) = number_of_envs(monitor_env.env)
function reset!(monitor_env::MonitorWrapperEnv)
    DRiL.reset!(monitor_env.env)
    #dont count the current episodes to the stats, since they are manually stopped
    monitor_env.current_episode_lengths .= 0
    monitor_env.current_episode_returns .= 0
    nothing
end

function step!(monitor_env::MonitorWrapperEnv, action)
    rewards, terminateds, truncateds, infos = step!(monitor_env.env, action)
    monitor_env.current_episode_returns .+= rewards
    monitor_env.current_episode_lengths .+= 1
    dones = terminateds .| truncateds
    for (i, done) in enumerate(dones)
        if done
            push!(monitor_env.episode_stats.episode_returns, monitor_env.current_episode_returns[i])
            push!(monitor_env.episode_stats.episode_lengths, monitor_env.current_episode_lengths[i])
            infos[i]["episode"] = Dict("r" => monitor_env.current_episode_returns[i], "l" => monitor_env.current_episode_lengths[i])
            monitor_env.current_episode_returns[i] = 0
            monitor_env.current_episode_lengths[i] = 0
        end
    end
    return rewards, terminateds, truncateds, infos
end

unwrap(env::MonitorWrapperEnv) = env.env

function log_stats(env::MonitorWrapperEnv, logger::TensorBoardLogger.TBLogger)
    if length(env.episode_stats.episode_returns) > 0
        log_value(logger, "env/ep_rew_mean", mean(env.episode_stats.episode_returns))
        log_value(logger, "env/ep_len_mean", mean(env.episode_stats.episode_lengths))
    end
    nothing
end
function log_stats(env::AbstractParallellEnvWrapper, logger::AbstractLogger)
    log_stats(unwrap(env), logger)
end
function unwrap_all(env::AbstractEnv)
    wrapped = true
    while wrapped
        env = unwrap(env)
        wrapped = is_wrapper(env)
    end
    return env
end

function observation_space(env::AbstractParallelEnv)
    return observation_space(env.envs[1])
end

function action_space(env::AbstractParallelEnv)
    return action_space(env.envs[1])
end

struct MultiThreadedParallelEnv{E<:AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}
    function MultiThreadedParallelEnv(envs::Vector{E}) where E<:AbstractEnv
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    @threads for sub_env in env.envs
        reset!(sub_env)
    end
    nothing
end

#TODO: check if this typing is correct
function observe(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    observations = Vector{Array{eltype(observation_space(env)),length(size(observation_space(env)))}}(undef, length(env.envs))
    @threads for i in 1:length(env.envs)
        observations[i] = observe(env.envs[i])
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

function act!(env::MultiThreadedParallelEnv{E}, actions::AbstractVector) where E<:AbstractEnv
    @assert length(actions) == length(env.envs) "Number of actions ($(length(actions))) must match number of environments ($(length(env.envs)))"

    T = eltype(observation_space(env))
    rewards = Vector{T}(undef, length(env.envs))
    terminateds = Vector{Bool}(undef, length(env.envs))
    truncateds = Vector{Bool}(undef, length(env.envs))
    infos = Vector{Dict{String,Any}}(undef, length(env.envs))

    @threads for i in 1:length(env.envs)
        rewards[i] = act!(env.envs[i], actions[i])

        # Capture termination status BEFORE reset
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

struct BroadcastedParallelEnv{E<:AbstractEnv} <: AbstractParallelEnv
    envs::Vector{E}

    function BroadcastedParallelEnv(envs::Vector{E}) where E<:AbstractEnv
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    reset!.(env.envs)
    nothing
end

function observe(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return observe.(env.envs)
end

function terminated(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return terminated.(env.envs)
end

function truncated(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return truncated.(env.envs)
end

function action_space(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return action_space(env.envs[1])
end

function observation_space(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return observation_space(env.envs[1])
end

function get_info(env::BroadcastedParallelEnv{E}) where E<:AbstractEnv
    return get_info.(env.envs)
end

function act!(env::BroadcastedParallelEnv{E}, actions::AbstractVector) where E<:AbstractEnv
    @assert length(actions) == length(env.envs) "Number of actions ($(length(actions))) must match number of environments ($(length(env.envs)))"

    # Use broadcasting for the main operations
    rewards = act!.(env.envs, actions)

    # Capture termination status BEFORE reset
    terminateds = terminated.(env.envs)
    truncateds = truncated.(env.envs)

    # Update infos using broadcasting
    infos = get_info.(env.envs)

    # Handle terminal observations for truncated environments
    for i in 1:length(env.envs)
        if truncateds[i]
            infos[i]["terminal_observation"] = observe(env.envs[i])
        end

        if terminateds[i] || truncateds[i]
            reset!(env.envs[i])
        end
    end

    return rewards, terminateds, truncateds, infos
end

number_of_envs(env::BroadcastedParallelEnv) = length(env.envs)
unwrap_all(env::BroadcastedParallelEnv) = env.envs

struct MultiAgentParallelEnv{E<:AbstractParallelEnv} <: AbstractParallelEnv
    envs::Vector{E}
    env_counts::Vector{Int}  # Number of sub-envs in each parallel env
    total_envs::Int          # Sum of all env_counts

    function MultiAgentParallelEnv(envs::Vector{E}) where E<:AbstractParallelEnv
        @assert !isempty(envs) "Must provide at least one parallel environment"

        # All sub-environments must have the same observation and action spaces
        @assert all(env -> isequal(observation_space(env), observation_space(envs[1])), envs) "All sub-environments must have the same observation space"
        @assert all(env -> isequal(action_space(env), action_space(envs[1])), envs) "All sub-environments must have the same action space"

        env_counts = [number_of_envs(env) for env in envs]
        total_envs = sum(env_counts)

        return new{E}(envs, env_counts, total_envs)
    end
end

function reset!(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    reset!.(env.envs)
    nothing
end

function observe(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    observations = observe.(env.envs)
    stacked_observations = vcat(observations...)
    return stacked_observations
end

function terminated(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    # all_terminated = Bool[]
    batch_terminated = terminated.(env.envs)
    stacked_terminated = vcat(batch_terminated...)
    return stacked_terminated
end

function truncated(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    batch_truncated = truncated.(env.envs)
    stacked_truncated = vcat(batch_truncated...)
    return stacked_truncated
end

function get_info(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    batch_infos = get_info.(env.envs)
    stacked_infos = vcat(batch_infos...)
    return stacked_infos
end

function act!(env::MultiAgentParallelEnv{E}, actions::AbstractVector) where E<:AbstractParallelEnv
    @assert length(actions) == number_of_envs(env) "Number of actions ($(length(actions))) must match total number of environments ($(number_of_envs(env)))"

    # Split actions into chunks for each sub-parallel-env
    idxs = cumsum([1; env.env_counts])
    chunk_indices = [idxs[i]:idxs[i+1]-1 for i in 1:length(env.envs)]
    action_chunks = [actions[chunk_indices[i]] for i in 1:length(env.envs)]

    # Execute actions on each sub-parallel-env and collect results
    stacked_rewards = Vector{Float32}(undef, number_of_envs(env))
    stacked_terminated = Vector{Bool}(undef, number_of_envs(env))
    stacked_truncated = Vector{Bool}(undef, number_of_envs(env))
    stacked_infos = Vector{Dict{String,Any}}(undef, number_of_envs(env))

    @threads for i in eachindex(env.envs)
        rewards, terminateds, truncateds, infos = act!(env.envs[i], action_chunks[i])
        stacked_rewards[chunk_indices[i]] .= rewards
        stacked_terminated[chunk_indices[i]] .= terminateds
        stacked_truncated[chunk_indices[i]] .= truncateds
        stacked_infos[chunk_indices[i]] .= infos
    end

    # batched_rewards = getindex.(all_returns, 1)
    # batched_terminated = getindex.(all_returns, 2)
    # batched_truncated = getindex.(all_returns, 3)
    # batched_infos = getindex.(all_returns, 4)

    # stacked_rewards = vcat(batched_rewards...)
    # stacked_terminated = vcat(batched_terminated...)
    # stacked_truncated = vcat(batched_truncated...)
    # stacked_infos = vcat(batched_infos...)

    return stacked_rewards, stacked_terminated, stacked_truncated, stacked_infos
end

function observation_space(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    return observation_space(env.envs[1])
end

function action_space(env::MultiAgentParallelEnv{E}) where E<:AbstractParallelEnv
    return action_space(env.envs[1])
end

number_of_envs(env::MultiAgentParallelEnv) = env.total_envs

function Random.seed!(env::MultiAgentParallelEnv, seed::Integer)
    for (i, sub_env) in enumerate(env.envs)
        Random.seed!(sub_env, seed + i - 1)
    end
    return env
end
unwrap_all(env::MultiAgentParallelEnv) = env.envs

struct ScalingWrapperEnv{E<:AbstractEnv,O<:AbstractSpace,A<:AbstractSpace} <: AbstractEnvWrapper{E}
    env::E
    observation_space::O
    action_space::A
    orig_observation_space::O
    orig_action_space::A
    # Pre-computed scaling factors for observations
    obs_scale_factor::Array{eltype(O)}
    obs_offset::Array{eltype(O)}
    # Pre-computed scaling factors for actions  
    act_scale_factor::Array{eltype(A)}
    act_offset::Array{eltype(A)}
end

function ScalingWrapperEnv(env::E) where {E<:AbstractEnv}
    orig_obs_space = observation_space(env)
    orig_act_space = action_space(env)

    return ScalingWrapperEnv(env, orig_obs_space, orig_act_space)
end

function ScalingWrapperEnv(env::E, original_obs_space::Box, original_act_space::Box) where {E<:AbstractEnv}
    # Create new observation space with bounds [-1, 1]
    T_obs = eltype(original_obs_space)
    T_act = eltype(original_act_space)

    scaled_obs_space = @set original_obs_space.low = -1 * ones(T_obs, size(original_obs_space.low))
    scaled_obs_space = @set scaled_obs_space.high = 1 * ones(T_obs, size(original_obs_space.high))

    # Create new action space with bounds [-1, 1]
    scaled_act_space = @set original_act_space.low = -1 * ones(T_act, size(original_act_space.low))
    scaled_act_space = @set scaled_act_space.high = 1 * ones(T_act, size(original_act_space.high))

    # Pre-compute scaling factors for observations: scale = 2 / (high - low), offset = low
    obs_range = original_obs_space.high .- original_obs_space.low
    obs_scale_factor = 2 ./ obs_range
    obs_offset = original_obs_space.low

    # Pre-compute scaling factors for actions
    act_range = original_act_space.high .- original_act_space.low
    act_scale_factor = 2 ./ act_range
    act_offset = original_act_space.low

    return ScalingWrapperEnv{E,Box,Box}(env, scaled_obs_space, scaled_act_space, original_obs_space, original_act_space,
        obs_scale_factor, obs_offset, act_scale_factor, act_offset)
end
#TODO:document/fix unwrap
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

# Fast in-place scaling functions using pre-computed factors
# PERFORMANCE NOTES:
# - Use the in-place versions (scale_observation!, unscale_action!, etc.) when possible
# - Pre-allocate output buffers and reuse them across calls
# - Scaling factors are pre-computed once during construction to avoid repeated calculations
# - All operations use @. macro for vectorized, allocation-free broadcasting
@inline function scale!(input, scale_factor, offset)
    @. input = (input - offset) * scale_factor - 1
    return nothing
end

@inline function unscale!(input, scale_factor, offset)
    @. input = (input + 1) / scale_factor + offset
    return nothing
end

# Allocating versions for compatibility
function scale_observation!(observation, env::ScalingWrapperEnv{E,Box,Box}) where E
    scale!(observation, env.obs_scale_factor, env.obs_offset)
    return nothing
end

function unscale_observation!(observation, env::ScalingWrapperEnv{E,Box,Box}) where E
    unscale!(observation, env.obs_scale_factor, env.obs_offset)
    return nothing
end



function observe(env::ScalingWrapperEnv{E,Box,Box}) where E
    orig_obs = observe(env.env)
    # Scale observation from original space to [-1, 1] using pre-computed factors
    scale_observation!(orig_obs, env)
    return orig_obs
end


function scale_action!(action, env::ScalingWrapperEnv{E,Box,Box}) where E
    scale!(action, env.act_scale_factor, env.act_offset)
    return nothing
end

function unscale_action!(action, env::ScalingWrapperEnv{E,Box,Box}) where E
    unscale!(action, env.act_scale_factor, env.act_offset)
    return nothing
end

function act!(env::ScalingWrapperEnv{E,Box,Box}, action) where E
    unscale_action!(action, env)
    return act!(env.env, action)
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
    Random.seed!(env::AbstractParallelEnv, seed::Integer)

Seed all sub-environments in a parallel environment with incremented seeds.
Each sub-environment gets seeded with `seed + i - 1` where `i` is the environment index.
"""
function Random.seed!(env::AbstractParallelEnv, seed::Integer)
    for (i, sub_env) in enumerate(env.envs)
        Random.seed!(sub_env, seed + i - 1)
    end
    return env
end

"""
    Random.seed!(env::BroadcastedParallelEnv, seed::Integer)

Seed all sub-environments in a broadcasted parallel environment with incremented seeds.
Each sub-environment gets seeded with `seed + i - 1` where `i` is the environment index.
"""
function Random.seed!(env::BroadcastedParallelEnv, seed::Integer)
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

struct NormalizeWrapperEnv{E<:AbstractParallelEnv,T<:AbstractFloat} <: AbstractParallelEnvWrapper{E}
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
) where {E<:AbstractParallelEnv,T<:AbstractFloat}

    obs_space = observation_space(env)
    n_envs = number_of_envs(env)

    # Initialize running statistics
    obs_rms = RunningMeanStd(T, size(obs_space))
    ret_rms = RunningMeanStd(T, ())
    returns = zeros(T, n_envs)

    # Initialize cache arrays
    old_obs = Array{T}(undef, size(obs_space)..., n_envs)
    old_rewards = Vector{T}(undef, n_envs)

    return NormalizeWrapperEnv{E,T}(env, obs_rms, ret_rms, returns, training, norm_obs, norm_reward,
        clip_obs, clip_reward, gamma, epsilon, old_obs, old_rewards)
end
DRiL.unwrap(env::NormalizeWrapperEnv) = env.env

# Convenience constructor
function NormalizeWrapperEnv(env::E; kwargs...) where {E<:AbstractParallelEnv}
    return NormalizeWrapperEnv{E,Float32}(env; kwargs...)
end

# Forward basic properties
observation_space(env::NormalizeWrapperEnv) = observation_space(env.env)
action_space(env::NormalizeWrapperEnv) = action_space(env.env)
number_of_envs(env::NormalizeWrapperEnv) = number_of_envs(env.env)

function reset!(env::NormalizeWrapperEnv{E,T}) where {E,T}
    reset!(env.env)
    obs = observe(env.env)

    # Store original observations BEFORE normalization
    #should we also store rewards or something?
    env.old_obs .= batch(obs, observation_space(env))
    env.returns .= zero(T)

    return nothing
end

function observe(env::NormalizeWrapperEnv{E,T}) where {E,T}
    obs = observe(env.env)

    # Store original observations and rewards for access
    env.old_obs .= batch(obs, observation_space(env))

    # Update observation statistics if in training mode
    if env.training && env.norm_obs
        obs_batch = batch(obs, observation_space(env))
        update!(env.obs_rms, obs_batch)
    end
    normalize_obs!.(obs, Ref(env))
    return obs
end

function act!(env::NormalizeWrapperEnv{E,T}, actions::AbstractVector) where {E,T}
    rewards, terminateds, truncateds, infos = act!(env.env, actions)
    env.old_rewards .= rewards

    # Update reward statistics and normalize
    if env.training && env.norm_reward
        update_reward_stats!(env, rewards)
    end
    normalize_rewards!(rewards, env)

    # Reset returns for terminated environments
    dones = terminateds .| truncateds
    for i in findall(dones)
        env.returns[i] = zero(T)
    end

    # Normalize terminal observations in infos
    for i in findall(truncateds)
        if haskey(infos[i], "terminal_observation")
            term_obs = infos[i]["terminal_observation"]
            normalize_obs!(term_obs, env)
            infos[i]["terminal_observation"] = term_obs
        end
    end

    return rewards, terminateds, truncateds, infos
end

function update_reward_stats!(env::NormalizeWrapperEnv, rewards::Vector{T}) where T<:AbstractFloat
    env.returns .= env.returns .* env.gamma .+ rewards
    # Update return statistics (single value, so we reshape for consistency)
    update!(env.ret_rms, reshape(env.returns, 1, length(env.returns)))
end

function normalize_obs!(obs, env::NormalizeWrapperEnv)
    if !env.norm_obs
        return obs
    end

    # Normalize using running statistics
    @. obs = (obs .- env.obs_rms.mean) ./ sqrt.(env.obs_rms.var .+ env.epsilon)
    clamp!(obs, -env.clip_obs, env.clip_obs)
    return nothing
end

function normalize_rewards!(rewards, env::NormalizeWrapperEnv)
    if !env.norm_reward
        return rewards
    end

    # Normalize rewards using return statistics
    @. rewards = rewards ./ sqrt(env.ret_rms.var[1] + env.epsilon)
    clamp!(rewards, -env.clip_reward, env.clip_reward)
    return nothing
end

function unnormalize_obs!(obs, env::NormalizeWrapperEnv)
    if !env.norm_obs
        return obs
    end
    @. obs = obs .* sqrt.(env.obs_rms.var .+ env.epsilon) .+ env.obs_rms.mean
    return nothing
end

function unnormalize_rewards!(rewards, env::NormalizeWrapperEnv)
    if !env.norm_reward
        return rewards
    end
    @. rewards = rewards .* sqrt(env.ret_rms.var[1] + env.epsilon)
    return nothing
end

# Get original (unnormalized) observations and rewards
get_original_obs(env::NormalizeWrapperEnv) = eachslice(env.old_obs, dims=ndims(env.old_obs))
get_original_rewards(env::NormalizeWrapperEnv) = env.old_rewards

# Forward other methods
terminated(env::NormalizeWrapperEnv) = terminated(env.env)
truncated(env::NormalizeWrapperEnv) = truncated(env.env)
function get_info(env::NormalizeWrapperEnv)
    infos = get_info(env.env)
    terminateds = terminated(env.env)
    truncateds = truncated(env.env)
    dones = terminateds .| truncateds

    for i in findall(dones)
        if haskey(infos[i], "terminal_observation")
            term_obs = infos[i]["terminal_observation"]
            normalize_obs!(term_obs, env)
            infos[i]["terminal_observation"] = term_obs
        end
    end
    return infos
end

function Random.seed!(env::NormalizeWrapperEnv, seed::Integer)
    Random.seed!(env.env, seed)
    return env
end

# Training mode control
set_training(env::AbstractEnv, ::Bool) = env #default to no-op
#TODO: fix/doc this
is_training(env::AbstractEnv) = true
set_training(env::NormalizeWrapperEnv{E,T}, training::Bool) where {E,T} = @set env.training = training
is_training(env::NormalizeWrapperEnv{E,T}) where {E,T} = env.training

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
function load_normalization_stats!(env::NormalizeWrapperEnv{E,T}, filepath::String) where {E,T<:AbstractFloat}
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

struct MonitorWrapperEnv{E<:AbstractParallelEnv,T} <: AbstractParallelEnvWrapper{E} where T<:AbstractFloat
    env::E
    current_episode_lengths::Vector{Int}
    current_episode_returns::Vector{T}
    episode_stats::EpisodeStats{T}
end

function MonitorWrapperEnv(env::E, stats_window::Int=100) where E<:AbstractParallelEnv
    T = eltype(observation_space(env))
    return MonitorWrapperEnv{E,T}(
        env,
        zeros(Int, number_of_envs(env)),
        zeros(T, number_of_envs(env)),
        EpisodeStats{T}(stats_window)
    )
end

#TODO clean up this so its not necessary to forward all the methods
observe(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = observe(monitor_env.env)
terminated(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = terminated(monitor_env.env)
truncated(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = truncated(monitor_env.env)
get_info(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = get_info(monitor_env.env)
action_space(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = action_space(monitor_env.env)
observation_space(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = observation_space(monitor_env.env)
number_of_envs(monitor_env::MonitorWrapperEnv{E,T}) where {E,T} = number_of_envs(monitor_env.env)

function reset!(monitor_env::MonitorWrapperEnv{E,T}) where {E,T}
    DRiL.reset!(monitor_env.env)
    #dont count the current episodes to the stats, since they are manually stopped
    monitor_env.current_episode_lengths .= 0
    monitor_env.current_episode_returns .= 0
    nothing
end

function act!(monitor_env::MonitorWrapperEnv{E,T}, actions::AbstractVector) where {E,T}
    rewards, terminateds, truncateds, infos = act!(monitor_env.env, actions)

    monitor_env.current_episode_returns .+= rewards
    monitor_env.current_episode_lengths .+= 1
    dones = terminateds .| truncateds

    for i in findall(dones)
        push!(monitor_env.episode_stats.episode_returns, monitor_env.current_episode_returns[i])
        push!(monitor_env.episode_stats.episode_lengths, monitor_env.current_episode_lengths[i])
        infos[i]["episode"] = Dict("r" => monitor_env.current_episode_returns[i], "l" => monitor_env.current_episode_lengths[i])
        monitor_env.current_episode_returns[i] = 0
        monitor_env.current_episode_lengths[i] = 0
    end

    return rewards, terminateds, truncateds, infos
end

unwrap(env::MonitorWrapperEnv) = env.env

function log_stats(env::MonitorWrapperEnv{E,T}, logger::TensorBoardLogger.TBLogger) where {E,T}
    if length(env.episode_stats.episode_returns) > 0
        log_value(logger, "env/ep_rew_mean", mean(env.episode_stats.episode_returns))
        log_value(logger, "env/ep_len_mean", mean(env.episode_stats.episode_lengths))
    end
    nothing
end

function log_stats(env::AbstractParallelEnvWrapper, logger::AbstractLogger)
    log_stats(unwrap(env), logger)
end

# Fallback for environments without monitoring
log_stats(env::AbstractParallelEnv, logger::AbstractLogger) = nothing

# ==============================================================================
# Show methods for nice environment display
# ==============================================================================

# MultiThreadedParallelEnv show methods
function Base.show(io::IO, env::MultiThreadedParallelEnv{E}) where E
    print(io, "MultiThreadedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiThreadedParallelEnv{E}) where E
    println(io, "MultiThreadedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)
    print(io, "  environments: ")
    show(io, env.envs[1])
end

# BroadcastedParallelEnv show methods
function Base.show(io::IO, env::BroadcastedParallelEnv{E}) where E
    print(io, "BroadcastedParallelEnv{", E, "}(", length(env.envs), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::BroadcastedParallelEnv{E}) where E
    println(io, "BroadcastedParallelEnv{", E, "}")
    println(io, "  - Number of environments: ", length(env.envs))
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)
    print(io, "  environments: ")
    show(io, env.envs[1])
end

# MultiAgentParallelEnv show methods
function Base.show(io::IO, env::MultiAgentParallelEnv{E}) where E
    print(io, "MultiAgentParallelEnv{", E, "}(", length(env.envs), " parallel envs, ", env.total_envs, " total envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MultiAgentParallelEnv{E}) where E
    println(io, "MultiAgentParallelEnv{", E, "}")
    println(io, "  - Number of parallel environments: ", length(env.envs))
    println(io, "  - Total environments: ", env.total_envs)
    println(io, "  - Environment counts per parallel env: ", env.env_counts)
    obs_space = observation_space(env)
    act_space = action_space(env)
    println(io, "  - Observation space: ", obs_space)
    println(io, "  - Action space: ", act_space)

    for (i, sub_env) in enumerate(env.envs)
        println(io, "  - Parallel env ", i, " (", env.env_counts[i], " envs): ")
        show(io, sub_env)
        if i < length(env.envs)
            println(io)
        end
    end
end

# ScalingWrapperEnv show methods  
function Base.show(io::IO, env::ScalingWrapperEnv{E,O,A}) where {E,O,A}
    print(io, "ScalingWrapperEnv{", E, "}")
end

function Base.show(io::IO, ::MIME"text/plain", env::ScalingWrapperEnv{E,O,A}) where {E,O,A}
    println(io, "ScalingWrapperEnv{", E, "}")
    println(io, "  - Scaled observation bounds: [-1, 1]")
    println(io, "  - Scaled action bounds: [-1, 1]")
    println(io, "  - Original observation space: ", env.orig_observation_space)
    println(io, "  - Original action space: ", env.orig_action_space)
    print(io, "  wrapped environment: ")
    show(io, env.env)
end

# NormalizeWrapperEnv show methods
function Base.show(io::IO, env::NormalizeWrapperEnv{E,T}) where {E,T}
    print(io, "NormalizeWrapperEnv{", E, ",", T, "}(", number_of_envs(env), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::NormalizeWrapperEnv{E,T}) where {E,T}
    println(io, "NormalizeWrapperEnv{", E, ",", T, "}")
    println(io, "  - Training mode: ", env.training)
    println(io, "  - Normalize observations: ", env.norm_obs)
    println(io, "  - Normalize rewards: ", env.norm_reward)
    println(io, "  - Observation clip: ±", env.clip_obs)
    println(io, "  - Reward clip: ±", env.clip_reward)
    println(io, "  - Discount factor (γ): ", env.gamma)
    println(io, "  - Epsilon: ", env.epsilon)

    if env.obs_rms.count > 0
        println(io, "  - Observation stats (n=", env.obs_rms.count, "):")
        println(io, "    • Mean: ", round.(env.obs_rms.mean, digits=3))
        println(io, "    • Std: ", round.(sqrt.(env.obs_rms.var .+ env.epsilon), digits=3))
    else
        println(io, "  - Observation stats: Not initialized")
    end

    if env.ret_rms.count > 0
        println(io, "  - Return stats (n=", env.ret_rms.count, "):")
        println(io, "    • Std: ", round(sqrt(env.ret_rms.var[1] + env.epsilon), digits=3))
    else
        println(io, "  - Return stats: Not initialized")
    end

    print(io, "  wrapped environment: ")
    show(io, env.env)
end

# MonitorWrapperEnv show methods
function Base.show(io::IO, env::MonitorWrapperEnv{E,T}) where {E,T}
    print(io, "MonitorWrapperEnv{", E, ",", T, "}(", number_of_envs(env), " envs)")
end

function Base.show(io::IO, ::MIME"text/plain", env::MonitorWrapperEnv{E,T}) where {E,T}
    println(io, "MonitorWrapperEnv{", E, ",", T, "}")
    println(io, "  - Number of environments: ", number_of_envs(env))
    println(io, "  - Stats window size: ", env.episode_stats.episode_returns.capacity)

    if length(env.episode_stats.episode_returns) > 0
        println(io, "  - Episode statistics (", length(env.episode_stats.episode_returns), " episodes):")
        println(io, "    • Mean return: ", round(mean(env.episode_stats.episode_returns), digits=3))
        println(io, "    • Mean length: ", round(mean(env.episode_stats.episode_lengths), digits=1))
        println(io, "    • Return range: [", round(minimum(env.episode_stats.episode_returns), digits=3),
            ", ", round(maximum(env.episode_stats.episode_returns), digits=3), "]")
    else
        println(io, "  - Episode statistics: No completed episodes")
    end

    # Show current episode progress
    any_active = any(x -> x > 0, env.current_episode_lengths)
    if any_active
        active_envs = sum(x -> x > 0, env.current_episode_lengths)
        max_len = maximum(env.current_episode_lengths)
        max_ret = maximum(env.current_episode_returns)
        println(io, "  - Current episodes: ", active_envs, " active, max length: ", max_len,
            ", max return: ", round(max_ret, digits=3))
    else
        println(io, "  - Current episodes: None active")
    end

    print(io, "  wrapped environment: ")
    show(io, env.env)
end


function observation_space(env::AbstractParallellEnv)
    return observation_space(env.envs[1])
end

function action_space(env::AbstractParallellEnv)
    return action_space(env.envs[1])
end

struct MultiThreadedParallelEnv{E<:AbstractEnv} <: AbstractParallellEnv
    envs::Vector{E}
    n_envs::Int
    function MultiThreadedParallelEnv(envs::Vector{E}) where E<:AbstractEnv
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> observation_space(env) == observation_space(envs[1]), envs) "All environments must have the same observation space"
        @assert all(env -> action_space(env) == action_space(envs[1]), envs) "All environments must have the same action space"
        return new{E}(envs, length(envs))
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

number_of_envs(env::MultiThreadedParallelEnv) = env.n_envs

struct ScalingWrapperEnv{E<:AbstractEnv,O<:AbstractSpace,A<:AbstractSpace} <: AbstractEnv
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
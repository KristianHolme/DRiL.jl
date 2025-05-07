mutable struct MultiThreadedParallelEnv{E<:AbstractEnv} <: AbstractParallellEnv
    envs::Vector{E}
    function MultiThreadedParallelEnv(envs::Vector{E}) where E<:AbstractEnv
        @assert all(env -> typeof(env) == E, envs) "All environments must be of the same type"
        @assert all(env -> observation_space(env) == observation_space(envs[1]), envs) "All environments must have the same observation space"
        @assert all(env -> action_space(env) == action_space(envs[1]), envs) "All environments must have the same action space"
        return new{E}(envs)
    end
end

function reset!(env::MultiThreadedParallelEnv{E}, rng::AbstractRNG=Random.default_rng()) where E<:AbstractEnv
    @threads for env in env.envs
        reset!(env, rng)
    end
    nothing
end

function observe(env::MultiThreadedParallelEnv{E}) where E<:AbstractEnv
    observation_space = observation_space(env.envs[1])
    type = observation_space.type
    observations = Array{type, length(observation_space.shape)+1}(undef, (observation_space.shape..., length(env.envs)))
    @threads for i in 1:length(env.envs)
        selectdim(observations, length(observation_space.shape)+1, i) .= observe(env.envs[i])
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
    type  = observation_space(env).type
    rewards = Vector{type}(undef, length(env.envs))
    terminateds = Vector{Bool}(undef, length(env.envs))
    truncateds = Vector{Bool}(undef, length(env.envs))
    infos = Vector{Dict{String, Any}}(undef, length(env.envs))
    action_dims = length(action_space(env).shape)
    @assert size(action) == (action_space(env).shape..., length(env.envs)) "Action must be of shape $((action_space(env).shape..., length(env.envs)))"
    @threads for i in 1:length(env.envs)
        @info "Stepping env $i"
        rewards[i] = act!(env.envs[i], selectdim(action, action_dims+1, i))
        @info "Reward: $rewards[i]"
        terminateds[i] = terminated(env.envs[i])
        truncateds[i] = truncated(env.envs[i])
        infos[i] = get_info(env.envs[i])
    end
    return rewards, terminateds, truncateds, infos
end
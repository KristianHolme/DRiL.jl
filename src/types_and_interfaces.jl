abstract type AbstractEnv end
abstract type AbstractParallellEnv <: AbstractEnv end

"""
reset!(env::AbstractEnv)
- reset the environment to the initial state
"""
function reset!(env::AbstractEnv)
    error("reset! not implemented for $(typeof(env))")
end

"""
act!(env::AbstractEnv, action)
- take an action in the environment, return reward
-mandatory for single envs
"""
function act!(env::AbstractEnv, action)
    error("act! not implemented for $(typeof(env))")
end

"""
observe(env::AbstractEnv)
- observe the environment, return observation
"""
function observe(env::AbstractEnv)
    error("observe not implemented for $(typeof(env))")
end

"""
terminated(env::AbstractEnv)
- check if the environment is terminated
"""
function terminated(env::AbstractEnv)
    error("terminated not implemented for $(typeof(env))")
end

"""
truncated(env::AbstractEnv)
- check if the environment is truncated
"""
function truncated(env::AbstractEnv)
    error("truncated not implemented for $(typeof(env))")
end

"""
action_space(env::AbstractEnv)
- return the action space of the environment
"""
function action_space(env::AbstractEnv)
    error("action_space not implemented for $(typeof(env))")
end

"""
observation_space(env::AbstractEnv)
- return the observation space of the environment
"""
function observation_space(env::AbstractEnv)
    error("observation_space not implemented for $(typeof(env))")
end

"""
get_info(env::AbstractEnv)
- return the info of the environment
-for single env
"""
function get_info(env::AbstractEnv)
    error("get_info not implemented for $(typeof(env))")
end

function number_of_envs(env::AbstractParallellEnv)::Int
    error("number_of_envs not implemented for $(typeof(env))")
end


abstract type AbstractAgent end

abstract type AbstractBuffer end

abstract type AbstractEnvWrapper{E<:AbstractEnv} <: AbstractEnv end
abstract type AbstractParallellEnvWrapper{E<:AbstractParallellEnv} <: AbstractParallellEnv end


is_wrapper(env::AbstractEnv) = env isa AbstractEnvWrapper
is_wrapper(env::AbstractParallellEnv) = env isa AbstractParallellEnvWrapper

function unwrap(env::AbstractEnvWrapper)
    error("unwrap not implemented for $(typeof(env))")
end

function unwrap(env::AbstractParallellEnvWrapper)
    error("unwrap not implemented for $(typeof(env))")
end

function log_stats(env::AbstractEnv, logger::AbstractLogger)
    nothing
end


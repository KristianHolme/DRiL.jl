abstract type AbstractEnv end
abstract type AbstractParallellEnv <: AbstractEnv end

"""
reset!(env::AbstractEnv)
- reset the environment to the initial state
"""
function reset!(env::AbstractEnv, rng::AbstractRNG=Random.default_rng()) end

"""
act!(env::AbstractEnv, action)
- take an action in the environment, return reward
"""
function act!(env::AbstractEnv, action) end #for single envs    

"""
step!(env::AbstractParallellEnv, action)
- take an action in the environment, return reward, terminated, truncated, info
- auto reset individual environments when they are terminated or truncated
    - when truncated, the last observation is given as info["terminal_observation"]
-for parallell envs
"""
function step!(env::AbstractParallellEnv, action) end #for parallel envs

"""
observe(env::AbstractEnv)
- observe the environment, return observation
"""
function observe(env::AbstractEnv) end

"""
terminated(env::AbstractEnv)
- check if the environment is terminated
"""
function terminated(env::AbstractEnv) end

"""
truncated(env::AbstractEnv)
- check if the environment is truncated
"""
function truncated(env::AbstractEnv) end

"""
action_space(env::AbstractEnv)
- return the action space of the environment
"""
function action_space(env::AbstractEnv) end

"""
observation_space(env::AbstractEnv)
- return the observation space of the environment
"""
function observation_space(env::AbstractEnv) end

"""
get_info(env::AbstractEnv)
- return the info of the environment
-for single env
"""
function get_info(env::AbstractEnv) end


abstract type AbstractAgent end

function get_action_and_value(agent::AbstractAgent, obs::AbstractArray) end

abstract type AbstractBuffer end


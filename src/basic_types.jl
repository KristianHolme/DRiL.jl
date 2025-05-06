abstract type AbstractEnv end
abstract type AbstractParallellEnv <: AbstractEnv end

function reset!(env::AbstractEnv) end
function act!(env::AbstractEnv, action) end
function step!(env::AbstractParallellEnv, action) end
function observe!(env::AbstractEnv) end
function terminated(env::AbstractEnv) end
function truncated(env::AbstractEnv) end
function action_space(env::AbstractEnv) end
function observation_space(env::AbstractEnv) end
function get_info(env::AbstractEnv) end

abstract type AbstractSpace end


abstract type AbstractAgent end

function get_action_and_value(agent::AbstractAgent, obs::AbstractArray) end



abstract type AbstractBuffer end


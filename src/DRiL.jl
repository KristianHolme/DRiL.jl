__precompile__(false) #disable precompilation for this module
module DRiL

using Random
using Statistics
using LinearAlgebra
using Base.Threads
include("basic_types.jl")
export AbstractEnv, AbstractAgent, AbstractBuffer
export reset!, act!, observe!, terminated, truncated, action_space, observation_space, get_info

include("spaces.jl") 
export AbstractSpace, AbstractBox, UniformBox

include("buffers.jl")
export Trajectory, RolloutBuffer

include("environments.jl")
export MultiThreadedParallelEnv
include("policies.jl")
export ActorCriticPolicy, AbstractPolicy
include("agents.jl")
end

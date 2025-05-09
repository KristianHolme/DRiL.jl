__precompile__(false) #disable precompilation for this module
module DRiL

using Base.Threads
using Distributions
using LinearAlgebra
using Lux
using Optimisers
using Random
using Statistics

include("basic_types.jl")
export AbstractEnv, AbstractAgent, AbstractBuffer
export reset!, act!, observe, terminated, truncated, action_space, observation_space, get_info

include("spaces.jl") 
export AbstractSpace, AbstractBox, UniformBox

include("policies.jl")
export ActorCriticPolicy, AbstractPolicy

include("buffers.jl")
export Trajectory, RolloutBuffer

include("environments.jl")
export MultiThreadedParallelEnv
include("agents.jl")
end

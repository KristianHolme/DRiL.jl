module DRiL

using Random
using Statistics
using LinearAlgebra

include("basic_types.jl")
export AbstractEnv, AbstractAgent, AbstractBuffer, AbstractSpace
export reset!, act!, observe!, terminated, truncated, action_space, observation_space, get_info

include("spaces.jl") 
export AbstractBox, UniformBox

include("buffers.jl")
export Trajectory

include("environments.jl")
# include("agents.jl")
end

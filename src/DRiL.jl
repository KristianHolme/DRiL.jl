__precompile__(false) #disable precompilation for this module
module DRiL

using Accessors
using Base.Threads
using ChainRulesCore
using Distributions
using LinearAlgebra
using Lux
using MLUtils
using Optimisers
using ProgressMeter
using Random
using Statistics
using TensorBoardLogger

include("basic_types.jl")
export AbstractEnv, AbstractAgent, AbstractBuffer
export reset!, act!, observe, terminated, truncated, action_space, observation_space, get_info

include("spaces.jl")
export AbstractSpace, AbstractBox, UniformBox

include("policies.jl")
export ActorCriticPolicy, AbstractPolicy

include("agents.jl")
export ActorCriticAgent, predict_actions, predict_values


include("buffers.jl")
export Trajectory, RolloutBuffer

include("environment_tools.jl")
export MultiThreadedParallelEnv, ScalingWrapperEnv

include("algorithms.jl")
export learn!, PPO

include("utils.jl")
export collect_trajectory

end

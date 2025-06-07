module DRiL

using Accessors
using Base.Threads
using ChainRulesCore
using ComponentArrays
using DataStructures
using Distributions
using LinearAlgebra
using Logging
using Lux
using MLUtils
using Optimisers
using ProgressMeter
using Random
using Statistics
using TensorBoardLogger
using FileIO
using JLD2

include("types_and_interfaces.jl")
export AbstractEnv, AbstractAgent, AbstractBuffer
export reset!, act!, observe, terminated, truncated, action_space, observation_space, get_info, number_of_envs

include("spaces.jl")
export AbstractSpace, AbstractBox, UniformBox, Box, Discrete

include("policies.jl")
export AbstractActorCriticPolicy, ActorCriticPolicy, ContinuousActorCriticPolicy, DiscreteActorCriticPolicy, AbstractPolicy, AbstractWeightInitializer, OrthogonalInitializer

include("agents.jl")
export ActorCriticAgent, predict_actions, predict_values

include("buffers.jl")
export Trajectory, RolloutBuffer

include("environment_tools.jl")
export MultiThreadedParallelEnv, BroadcastedParallelEnv, ScalingWrapperEnv, NormalizeWrapperEnv, RunningMeanStd
export save_normalization_stats, load_normalization_stats!, set_training, is_training
export get_original_obs, get_original_rewards, normalize_obs, normalize_reward, unnormalize_obs, unnormalize_reward
export MonitorWrapperEnv, EpisodeStats, is_wrapper, unwrap, unwrap_all

include("algorithms.jl")
export learn!, PPO

include("utils.jl")
export collect_trajectory

include("logging.jl")
export get_hparams

include("env_checker.jl")
export check_env

include("evaluation.jl")
export evaluate_agent

end

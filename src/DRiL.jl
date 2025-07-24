module DRiL

using Accessors
using Base.Threads
using ChainRulesCore
using ComponentArrays
using DataStructures
using LinearAlgebra
using Logging
using Lux
using MLUtils
using Optimisers
using ProgressMeter
using Reexport
using Random
using Statistics
using TensorBoardLogger
using FileIO
using JLD2

include("DRiLDistributions/DRiLDistributions.jl")
@reexport using .DRiLDistributions

include("types_and_interfaces.jl")
export AbstractEnv, AbstractParallelEnv, AbstractAgent, AbstractBuffer, AbstractAlgorithm
export AbstractEntropyTarget, FixedEntropyTarget, AutoEntropyTarget
export AbstractEntropyCoefficient, FixedEntropyCoefficient, AutoEntropyCoefficient
export reset!, act!, observe, terminated, truncated, action_space, observation_space, get_info, number_of_envs


include("spaces.jl")
export AbstractSpace, Box, Discrete

include("policies.jl")
export AbstractActorCriticPolicy, ActorCriticPolicy, ContinuousActorCriticPolicy, 
DiscreteActorCriticPolicy, AbstractPolicy, AbstractWeightInitializer, 
OrthogonalInitializer, QCritic, VCritic

include("agents.jl")
export ActorCriticAgent, predict_actions, predict_values, steps_taken

include("buffers.jl")
export Trajectory, RolloutBuffer

include("callbacks.jl")
export AbstractCallback, on_training_start, on_training_end, on_rollout_start, on_rollout_end, on_step

include("environment_tools.jl")
export MultiThreadedParallelEnv, BroadcastedParallelEnv, ScalingWrapperEnv, NormalizeWrapperEnv, RunningMeanStd
export save_normalization_stats, load_normalization_stats!, set_training, is_training
export get_original_obs, get_original_rewards, normalize_obs!, normalize_rewards!, unnormalize_obs!, unnormalize_rewards!
export MonitorWrapperEnv, EpisodeStats, is_wrapper, unwrap, unwrap_all
export MultiAgentParallelEnv

include("algorithms/ppo.jl")
export learn!, PPO

include("algorithms/sac.jl")
export SAC


include("utils.jl")
export collect_trajectory

include("logging.jl")
export get_hparams

include("env_checker.jl")
export check_env

include("evaluation.jl")
export evaluate_agent

end
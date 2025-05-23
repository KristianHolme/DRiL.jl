# Test module containing shared environments and policies for testing.
# This module will be evaluated once per Julia test process and made available to test items.
@testmodule SharedTestSetup begin
    using DRiL
    using Random
    using DRiL.Lux

    # Custom environment that gives a reward of 1.0 only at the final timestep of an episode.
    # Equivalent to CustomEnv in stable-baselines3 test_gae.py.
    mutable struct CustomEnv <: AbstractEnv
        max_steps::Int
        n_steps::Int
        observation_space::UniformBox
        action_space::UniformBox
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String,Any}
        rng::Random.AbstractRNG

        function CustomEnv(max_steps::Int=8, rng::Random.AbstractRNG=Random.Xoshiro())
            obs_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
            act_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
            new(max_steps, 0, obs_space, act_space, false, false, 0.0f0, Dict{String,Any}(), rng)
        end
    end

    DRiL.observation_space(env::CustomEnv) = env.observation_space
    DRiL.action_space(env::CustomEnv) = env.action_space
    DRiL.terminated(env::CustomEnv) = env._terminated
    DRiL.truncated(env::CustomEnv) = env._truncated
    DRiL.get_info(env::CustomEnv) = env._info

    function DRiL.reset!(env::CustomEnv)
        env.n_steps = 0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String,Any}()
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    function DRiL.act!(env::CustomEnv, action::AbstractArray)
        env.n_steps += 1

        # Reward of 1.0 only at the final step, 0.0 otherwise
        reward = (env.n_steps >= env.max_steps) ? 1.0f0 : 0.0f0

        # Episode terminates when max_steps is reached
        env._terminated = env.n_steps >= env.max_steps
        # To simplify GAE computation checks, we do not consider truncation here
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String,Any}()

        return reward
    end

    function DRiL.step!(env::CustomEnv, action::AbstractArray)
        reward = DRiL.act!(env, action)

        # Random next observation using env's RNG
        next_obs = rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0

        return next_obs, reward, env._terminated, env._truncated, env._info
    end

    function DRiL.observe(env::CustomEnv)
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    # Infinite horizon environment that gives reward of 1.0 at every step and never terminates.
    # Modified from SB3's InfiniteHorizonEnv to use UniformBox instead of discrete space.
    mutable struct InfiniteHorizonEnv <: AbstractEnv
        n_states::Int
        current_state::Float32
        observation_space::UniformBox
        action_space::UniformBox
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String,Any}
        rng::Random.AbstractRNG

        function InfiniteHorizonEnv(n_states::Int=4, rng::Random.AbstractRNG=Random.Xoshiro())
            # Use continuous observation space [0, n_states] to represent the states
            obs_space = UniformBox{Float32}(0.0f0, Float32(n_states), (1,))
            act_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
            new(n_states, 0.0f0, obs_space, act_space, false, false, 0.0f0, Dict{String,Any}(), rng)
        end
    end

    DRiL.observation_space(env::InfiniteHorizonEnv) = env.observation_space
    DRiL.action_space(env::InfiniteHorizonEnv) = env.action_space
    DRiL.terminated(env::InfiniteHorizonEnv) = env._terminated
    DRiL.truncated(env::InfiniteHorizonEnv) = env._truncated
    DRiL.get_info(env::InfiniteHorizonEnv) = env._info

    function DRiL.reset!(env::InfiniteHorizonEnv)
        env.current_state = 0.0f0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String,Any}()
        return [env.current_state]
    end

    function DRiL.act!(env::InfiniteHorizonEnv, action::AbstractArray)
        env.current_state = Float32((Int(env.current_state) + 1) % env.n_states)

        # Always gives reward of 1.0
        reward = 1.0f0

        # Never terminates or truncates
        env._terminated = false
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String,Any}()

        return reward
    end

    function DRiL.step!(env::InfiniteHorizonEnv, action::AbstractArray)
        reward = DRiL.act!(env, action)

        # Return current state as observation
        next_obs = [env.current_state]

        return next_obs, reward, env._terminated, env._truncated, env._info
    end

    function DRiL.observe(env::InfiniteHorizonEnv)
        return [env.current_state]
    end

    # Simple environment that gives a reward of 1.0 only at the final timestep of an episode.
    # This allows for analytical computation of GAE values for testing.
    mutable struct SimpleRewardEnv <: AbstractEnv
        max_steps::Int
        current_step::Int
        observation_space::UniformBox
        action_space::UniformBox
        _terminated::Bool
        _truncated::Bool
        _last_reward::Float32
        _info::Dict{String,Any}
        rng::Random.AbstractRNG

        function SimpleRewardEnv(max_steps::Int=8, rng::Random.AbstractRNG=Random.Xoshiro())
            obs_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
            act_space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
            new(max_steps, 0, obs_space, act_space, false, false, 0.0f0, Dict{String,Any}(), rng)
        end
    end

    DRiL.observation_space(env::SimpleRewardEnv) = env.observation_space
    DRiL.action_space(env::SimpleRewardEnv) = env.action_space
    DRiL.terminated(env::SimpleRewardEnv) = env._terminated
    DRiL.truncated(env::SimpleRewardEnv) = env._truncated
    DRiL.get_info(env::SimpleRewardEnv) = env._info

    function DRiL.reset!(env::SimpleRewardEnv)
        env.current_step = 0
        env._terminated = false
        env._truncated = false
        env._last_reward = 0.0f0
        env._info = Dict{String,Any}()
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    function DRiL.act!(env::SimpleRewardEnv, action::AbstractArray)
        env.current_step += 1

        # Reward of 1.0 only at the final step, 0.0 otherwise
        reward = (env.current_step >= env.max_steps) ? 1.0f0 : 0.0f0

        # Episode terminates when max_steps is reached
        env._terminated = env.current_step >= env.max_steps
        env._truncated = false
        env._last_reward = reward
        env._info = Dict{String,Any}()

        return reward
    end

    function DRiL.step!(env::SimpleRewardEnv, action::AbstractArray)
        reward = DRiL.act!(env, action)

        # Random next observation using env's RNG
        next_obs = rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0

        return next_obs, reward, env._terminated, env._truncated, env._info
    end

    function DRiL.observe(env::SimpleRewardEnv)
        return rand(env.rng, Float32, 2) .* 2.0f0 .- 1.0f0  # Use env's RNG
    end

    # Custom policy that returns constant values for predictable GAE testing.
    struct ConstantValuePolicy <: DRiL.AbstractActorCriticPolicy
        observation_space::UniformBox{Float32}
        action_space::UniformBox{Float32}
        constant_value::Float32
    end

    # Implement Lux interface functions
    function Lux.initialparameters(rng::AbstractRNG, policy::ConstantValuePolicy)
        # No learnable parameters needed - the constant value is just configuration
        return NamedTuple()
    end

    function Lux.initialstates(rng::AbstractRNG, policy::ConstantValuePolicy)
        # No states needed for constant policy
        return NamedTuple()
    end

    function Lux.parameterlength(policy::ConstantValuePolicy)
        # No parameters
        return 0
    end

    function Lux.statelength(policy::ConstantValuePolicy)
        # No states
        return 0
    end

    function DRiL.predict_values(policy::ConstantValuePolicy, observations::AbstractArray)
        batch_size = size(observations)[end]
        return fill(policy.constant_value, batch_size)
    end

    # Generic predict_values method that takes policy, observations, parameters, and states
    function DRiL.predict_values(policy::ConstantValuePolicy, observations::AbstractArray, ps, st)
        batch_size = size(observations)[end]
        return fill(policy.constant_value, batch_size), st
    end

    # Implement the main policy call function
    function (policy::ConstantValuePolicy)(obs::AbstractArray, ps, st; rng::AbstractRNG=Random.default_rng())
        batch_size = size(obs)[end]
        # Random actions in action space bounds
        actions = rand(rng, Float32, policy.action_space.shape..., batch_size) .* 2.0f0 .- 1.0f0
        values = fill(policy.constant_value, batch_size)
        logprobs = fill(0.0f0, batch_size)
        return actions, values, logprobs, st
    end

    # Implement predict function
    function DRiL.predict(policy::ConstantValuePolicy, obs::AbstractArray, ps, st; deterministic::Bool=false, rng::AbstractRNG=Random.default_rng())
        batch_size = size(obs)[end]
        actions = rand(rng, Float32, policy.action_space.shape..., batch_size) .* 2.0f0 .- 1.0f0
        return actions, st
    end

    # Implement evaluate_actions function  
    function DRiL.evaluate_actions(policy::ConstantValuePolicy, obs::AbstractArray, actions::AbstractArray, ps, st)
        batch_size = size(obs)[end]
        values = fill(policy.constant_value, batch_size)
        logprobs = fill(0.0f0, batch_size)
        entropy = fill(0.0f0, batch_size)
        return values, logprobs, entropy, st
    end

    # Helper function to compute expected GAE advantages analytically.
    function compute_expected_gae(rewards::Vector{T}, values::Vector{T}, gamma::T, gae_lambda::T;
        is_terminated::Bool=true, bootstrap_value::Union{Nothing,T}=nothing) where T<:AbstractFloat
        n = length(rewards)
        expected_advantages = zeros(T, n)

        # Last step calculation
        if is_terminated || isnothing(bootstrap_value)
            # No bootstrapping for terminated episodes
            expected_advantages[n] = rewards[n] - values[n]
        else
            # Bootstrap for truncated or rollout-limited trajectories
            expected_advantages[n] = rewards[n] + gamma * bootstrap_value - values[n]
        end

        # Backward pass through earlier steps
        for t in (n-1):-1:1
            delta = rewards[t] + gamma * values[t+1] - values[t]
            expected_advantages[t] = delta + gamma * gae_lambda * expected_advantages[t+1]
        end

        return expected_advantages
    end

    # Define AbstractEnvWrapper since it's not in the main codebase
    abstract type AbstractEnvWrapper <: AbstractEnv end

    # Environment wrapper to ensure consistent observations for testing.
    mutable struct ConstantObsWrapper <: AbstractEnvWrapper
        env::AbstractEnv
        constant_obs::Vector{Float32}

        function ConstantObsWrapper(env::AbstractEnv, obs::Vector{Float32})
            new(env, obs)
        end
    end

    # Forward all methods to wrapped environment
    DRiL.observation_space(wrapper::ConstantObsWrapper) = DRiL.observation_space(wrapper.env)
    DRiL.action_space(wrapper::ConstantObsWrapper) = DRiL.action_space(wrapper.env)
    DRiL.terminated(wrapper::ConstantObsWrapper) = DRiL.terminated(wrapper.env)
    DRiL.truncated(wrapper::ConstantObsWrapper) = DRiL.truncated(wrapper.env)
    DRiL.get_info(wrapper::ConstantObsWrapper) = DRiL.get_info(wrapper.env)

    function DRiL.reset!(wrapper::ConstantObsWrapper)
        DRiL.reset!(wrapper.env)
        return copy(wrapper.constant_obs)
    end

    function DRiL.act!(wrapper::ConstantObsWrapper, action::AbstractArray)
        return DRiL.act!(wrapper.env, action)
    end

    function DRiL.step!(wrapper::ConstantObsWrapper, action::AbstractArray)
        next_obs, reward, terminated, truncated, info = DRiL.step!(wrapper.env, action)
        # Return constant observation instead of the environment's observation
        return copy(wrapper.constant_obs), reward, terminated, truncated, info
    end

    function DRiL.observe(wrapper::ConstantObsWrapper)
        return copy(wrapper.constant_obs)
    end
end
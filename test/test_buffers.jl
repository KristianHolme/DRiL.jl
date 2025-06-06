using TestItems

@testitem "Buffer logprobs consistency" tags = [:buffers, :rollouts] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: logprobs are consistent after rollout
    pend_env() = PendulumEnv()
    env = MultiThreadedParallelEnv([pend_env() for _ in 1:4])
    policy = ActorCriticPolicy(DRiL.observation_space(env), DRiL.action_space(env))
    agent = ActorCriticAgent(policy; n_steps=8, batch_size=8, epochs=1, verbose=0)
    alg = PPO()
    n_steps = agent.n_steps
    n_envs = DRiL.number_of_envs(env)
    roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    for i in 1:10
        DRiL.collect_rollouts!(roll_buffer, agent, env)
        obs = roll_buffer.observations
        act = roll_buffer.actions
        logprobs = roll_buffer.logprobs
        ps = agent.train_state.parameters
        st = agent.train_state.states
        _, new_logprobs, _, _ = DRiL.evaluate_actions(policy, obs, act, ps, st)
        @test isapprox(vec(logprobs), vec(new_logprobs); atol=1e-5, rtol=1e-5)
    end
end

@testitem "Buffer reset functionality" tags = [:buffers] setup = [SharedTestSetup] begin
    using Random

    # Test buffer reset clears all data
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0], Float32[1.0])

    roll_buffer = RolloutBuffer(obs_space, act_space, 0.95f0, 0.99f0, 8, 2)

    # Fill buffer with some data
    roll_buffer.observations .= 1.0f0
    roll_buffer.actions .= 2.0f0
    roll_buffer.rewards .= 3.0f0
    roll_buffer.advantages .= 4.0f0
    roll_buffer.returns .= 5.0f0
    roll_buffer.logprobs .= 6.0f0
    roll_buffer.values .= 7.0f0

    # Reset buffer
    DRiL.reset!(roll_buffer)

    # Verify all arrays are zeroed
    @test all(iszero, roll_buffer.observations)
    @test all(iszero, roll_buffer.actions)
    @test all(iszero, roll_buffer.rewards)
    @test all(iszero, roll_buffer.advantages)
    @test all(iszero, roll_buffer.returns)
    @test all(iszero, roll_buffer.logprobs)
    @test all(iszero, roll_buffer.values)
end

@testitem "Buffer trajectory bootstrap handling" tags = [:buffers, :bootstrap] setup = [SharedTestSetup] begin
    using Random

    # Test bootstrap value handling for truncated episodes
    max_steps = 6
    gamma = 0.9f0
    gae_lambda = 0.8f0
    constant_value = 0.7f0
    bootstrap_value = 0.2f0

    # Create a simple trajectory manually to test bootstrap handling
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    # Create trajectory with known values
    traj = Trajectory(obs_space, act_space)

    # Add some dummy data
    for i in 1:max_steps
        push!(traj.observations, rand(Float32, 2))
        push!(traj.actions, rand(Float32, 2))
        push!(traj.rewards, i == max_steps ? 1.0f0 : 0.0f0)  # Reward only at end
        push!(traj.logprobs, 0.0f0)
        push!(traj.values, constant_value)
    end

    # Test terminated trajectory (no bootstrap)
    traj.terminated = true
    traj.truncated = false
    traj.bootstrap_value = nothing

    advantages_terminated = zeros(Float32, max_steps)
    DRiL.compute_advantages!(advantages_terminated, traj, gamma, gae_lambda)

    expected_terminated = SharedTestSetup.compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda; is_terminated=true
    )
    @test isapprox(advantages_terminated, expected_terminated, atol=1e-4)

    # Test truncated trajectory (with bootstrap)
    traj.terminated = false
    traj.truncated = true
    traj.bootstrap_value = bootstrap_value

    advantages_truncated = zeros(Float32, max_steps)
    DRiL.compute_advantages!(advantages_truncated, traj, gamma, gae_lambda)

    expected_truncated = SharedTestSetup.compute_expected_gae(
        traj.rewards, traj.values, gamma, gae_lambda;
        is_terminated=false, bootstrap_value=bootstrap_value
    )
    @test isapprox(advantages_truncated, expected_truncated, atol=1e-4)

    # Verify that bootstrapped case gives different results
    @test !isapprox(advantages_terminated, advantages_truncated, atol=1e-3)
end

@testitem "Buffer data integrity" tags = [:buffers, :integrity] setup = [SharedTestSetup] begin
    using Random

    # Test that buffer maintains data integrity during rollout collection
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])  # Match SimpleRewardEnv shape
    act_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])

    n_steps = 16
    n_envs = 2
    gamma = 0.99f0
    gae_lambda = 0.95f0

    roll_buffer = RolloutBuffer(obs_space, act_space, gae_lambda, gamma, n_steps, n_envs)

    # Create simple test environment
    env = MultiThreadedParallelEnv([SharedTestSetup.SimpleRewardEnv(8) for _ in 1:n_envs])
    env_obs_space = DRiL.observation_space(env)
    env_act_space = DRiL.action_space(env)
    @test isequal(env_obs_space, obs_space)
    @test isequal(env_act_space, act_space)


    policy = SharedTestSetup.ConstantValuePolicy(env_obs_space, env_act_space, 0.5f0)
    agent = ActorCriticAgent(policy; n_steps=n_steps, batch_size=16, epochs=1, verbose=0)

    # Collect rollouts
    DRiL.collect_rollouts!(roll_buffer, agent, env)

    # Verify buffer dimensions
    @test size(roll_buffer.observations) == (obs_space.shape..., n_steps * n_envs)
    @test size(roll_buffer.actions) == (act_space.shape..., n_steps * n_envs)
    @test length(roll_buffer.rewards) == n_steps * n_envs
    @test length(roll_buffer.advantages) == n_steps * n_envs
    @test length(roll_buffer.returns) == n_steps * n_envs
    @test length(roll_buffer.logprobs) == n_steps * n_envs
    @test length(roll_buffer.values) == n_steps * n_envs

    # Verify no NaN or Inf values
    @test all(isfinite, roll_buffer.rewards)
    @test all(isfinite, roll_buffer.advantages)
    @test all(isfinite, roll_buffer.returns)
    @test all(isfinite, roll_buffer.logprobs)
    @test all(isfinite, roll_buffer.values)

    # Verify returns = advantages + values relationship
    @test isapprox(roll_buffer.returns, roll_buffer.advantages .+ roll_buffer.values, atol=1e-5)
end

@testitem "RolloutBuffer with discrete actions" tags = [:buffers, :rollouts, :discrete] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: RolloutBuffer works correctly with discrete action spaces
    cartpole_env() = CartPoleEnv()
    env = MultiThreadedParallelEnv([cartpole_env() for _ in 1:4])
    policy = DiscreteActorCriticPolicy(DRiL.observation_space(env), DRiL.action_space(env))
    agent = ActorCriticAgent(policy; n_steps=8, batch_size=8, epochs=1, verbose=0)
    alg = PPO()
    
    n_steps = agent.n_steps
    n_envs = DRiL.number_of_envs(env)
    roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)

    # Test rollout collection
    DRiL.collect_rollouts!(roll_buffer, agent, env)
    
    # Check that actions are stored as 1-based indices (raw policy output)
    actions = roll_buffer.actions
    @test all(1 .<= actions .<= DRiL.action_space(env).n)  # Should be in [1, n]
    @test eltype(actions) <: Integer
    
    # Check observations are valid
    obs = roll_buffer.observations
    obs_space = DRiL.observation_space(env)
    @test size(obs) == (obs_space.shape..., n_steps, n_envs)
    @test eltype(obs) == Float32
    
    # Check that rewards are reasonable
    rewards = roll_buffer.rewards
    @test all(rewards .>= 0.0f0)  # CartPole gives positive rewards
    @test size(rewards) == (n_steps, n_envs)
    
    # Check that log probabilities are consistent
    logprobs = roll_buffer.logprobs
    values = roll_buffer.values
    @test size(logprobs) == (n_steps, n_envs)
    @test size(values) == (n_steps, n_envs)
    
    # Test action evaluation consistency
    ps = agent.train_state.parameters
    st = agent.train_state.states
    eval_values, eval_logprobs, entropy, _ = DRiL.evaluate_actions(policy, obs, actions, ps, st)
    
    @test isapprox(vec(values), vec(eval_values); atol=1e-5, rtol=1e-5)
    @test isapprox(vec(logprobs), vec(eval_logprobs); atol=1e-5, rtol=1e-5)
    @test all(entropy .>= 0.0f0)  # Entropy should be non-negative
end

@testitem "Discrete vs continuous buffer comparison" tags = [:buffers, :rollouts, :comparison] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: Compare buffer behavior between discrete and continuous action spaces
    
    # Discrete environment (CartPole)
    discrete_env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:2])
    discrete_policy = DiscreteActorCriticPolicy(DRiL.observation_space(discrete_env), DRiL.action_space(discrete_env))
    discrete_agent = ActorCriticAgent(discrete_policy; n_steps=4, batch_size=4, epochs=1, verbose=0)
    
    # Continuous environment (Pendulum)
    continuous_env = MultiThreadedParallelEnv([PendulumEnv() for _ in 1:2])
    continuous_policy = ContinuousActorCriticPolicy(DRiL.observation_space(continuous_env), DRiL.action_space(continuous_env))
    continuous_agent = ActorCriticAgent(continuous_policy; n_steps=4, batch_size=4, epochs=1, verbose=0)
    
    alg = PPO()
    
    # Create buffers
    discrete_buffer = RolloutBuffer(DRiL.observation_space(discrete_env), DRiL.action_space(discrete_env), alg.gae_lambda, alg.gamma, 4, 2)
    continuous_buffer = RolloutBuffer(DRiL.observation_space(continuous_env), DRiL.action_space(continuous_env), alg.gae_lambda, alg.gamma, 4, 2)
    
    # Collect rollouts
    DRiL.collect_rollouts!(discrete_buffer, discrete_agent, discrete_env)
    DRiL.collect_rollouts!(continuous_buffer, continuous_agent, continuous_env)
    
    # Test discrete actions are integers
    discrete_actions = discrete_buffer.actions
    @test eltype(discrete_actions) <: Integer
    @test all(1 .<= discrete_actions .<= 2)  # CartPole has 2 actions
    
    # Test continuous actions are floats
    continuous_actions = continuous_buffer.actions
    @test eltype(continuous_actions) <: AbstractFloat
    @test size(continuous_actions) == (1, 4, 2)  # (action_dim, n_steps, n_envs)
    
    # Test that both have same buffer structure otherwise
    @test size(discrete_buffer.observations) == size(continuous_buffer.observations)  # Both should be (4, n_steps, n_envs)
    @test size(discrete_buffer.rewards) == size(continuous_buffer.rewards)
    @test size(discrete_buffer.logprobs) == size(continuous_buffer.logprobs)
    @test size(discrete_buffer.values) == size(continuous_buffer.values)
    
    # Test that evaluation works for both
    discrete_ps = discrete_agent.train_state.parameters
    discrete_st = discrete_agent.train_state.states
    continuous_ps = continuous_agent.train_state.parameters
    continuous_st = continuous_agent.train_state.states
    
    # Discrete evaluation
    discrete_eval_values, discrete_eval_logprobs, discrete_entropy, _ = DRiL.evaluate_actions(
        discrete_policy, discrete_buffer.observations, discrete_buffer.actions, discrete_ps, discrete_st)
    
    # Continuous evaluation
    continuous_eval_values, continuous_eval_logprobs, continuous_entropy, _ = DRiL.evaluate_actions(
        continuous_policy, continuous_buffer.observations, continuous_buffer.actions, continuous_ps, continuous_st)
    
    # Test evaluation consistency
    @test isapprox(vec(discrete_buffer.values), vec(discrete_eval_values); atol=1e-5, rtol=1e-5)
    @test isapprox(vec(continuous_buffer.values), vec(continuous_eval_values); atol=1e-5, rtol=1e-5)
    @test isapprox(vec(discrete_buffer.logprobs), vec(discrete_eval_logprobs); atol=1e-5, rtol=1e-5)
    @test isapprox(vec(continuous_buffer.logprobs), vec(continuous_eval_logprobs); atol=1e-5, rtol=1e-5)
end

@testitem "Discrete action indexing in buffers" tags = [:buffers, :discrete, :indexing] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test: Verify that discrete action indexing is consistent throughout the pipeline
    
    # Create CartPole environment with 0-based action space
    env = MultiThreadedParallelEnv([CartPoleEnv() for _ in 1:2])
    policy = DiscreteActorCriticPolicy(DRiL.observation_space(env), DRiL.action_space(env))
    agent = ActorCriticAgent(policy; n_steps=4, batch_size=4, epochs=1, verbose=0)
    alg = PPO()
    
    # Create buffer
    roll_buffer = RolloutBuffer(DRiL.observation_space(env), DRiL.action_space(env), alg.gae_lambda, alg.gamma, 4, 2)
    
    # Collect rollouts
    DRiL.collect_rollouts!(roll_buffer, agent, env)
    
    # Actions in buffer should be 1-based (raw policy outputs before processing)
    stored_actions = roll_buffer.actions
    @test all(1 .<= stored_actions .<= 2)  # Should be 1 or 2 (1-based)
    
    # Test manual action prediction and processing
    obs = DRiL.observe(env)  # Get current observations
    ps = agent.train_state.parameters
    st = agent.train_state.states
    
    # Policy output (1-based)
    policy_actions, _, _, _ = policy(obs[:, 1:1], ps, st)  # Single observation
    @test 1 <= policy_actions <= 2  # Should be 1-based
    
    # Processed actions for environment (0-based)
    processed_actions, _ = predict(policy, obs[:, 1:1], ps, st)
    @test processed_actions ∈ DRiL.action_space(env)  # Should be 0 or 1
    @test processed_actions == policy_actions + (DRiL.action_space(env).start - 1)
    
    # Verify that evaluate_actions works with stored (1-based) actions
    eval_values, eval_logprobs, entropy, _ = DRiL.evaluate_actions(
        policy, roll_buffer.observations, stored_actions, ps, st)
    
    @test size(eval_values) == size(roll_buffer.values)
    @test size(eval_logprobs) == size(roll_buffer.logprobs)
    @test all(entropy .>= 0.0f0)
    
    # Test that the indexing conversion is correct
    action_space = DRiL.action_space(env)
    for stored_action in unique(stored_actions)
        processed = process_action(stored_action, action_space)
        @test processed ∈ action_space
        @test processed == stored_action + (action_space.start - 1)  # 1-based to 0-based
    end
end
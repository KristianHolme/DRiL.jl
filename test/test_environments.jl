using Test
using DRiL
using TestItems

@testset "environments.jl" begin
    # TODO: Add tests for environments
end

@testitem "CartPole environment construction" tags = [:environments, :cartpole, :construction] begin
    using ClassicControlEnvironments
    using Random

    # Test basic construction
    env = CartPoleEnv()
    @test env isa CartPoleEnv
    
    # Test with custom parameters
    problem_custom = CartPoleProblem(
        gravity=10.0f0,
        mass_cart=1.5f0,
        mass_pole=0.15f0,
        total_mass=1.65f0,
        length=0.6f0,
        polemass_length=0.09f0,
        force_mag=12.0f0,
        tau=0.025f0,
        theta_threshold_radians=15.0f0 * π / 180.0f0,
        x_threshold=3.0f0,
        reward_style=:sutton_barto
    )
    env_custom = CartPoleEnv(problem_custom)
    @test env_custom.problem.force_mag == 12.0f0
    @test env_custom.problem.reward_style == :sutton_barto
    
    # Test with different reward styles
    env_standard = CartPoleEnv(CartPoleProblem(reward_style=:standard))
    env_sutton = CartPoleEnv(CartPoleProblem(reward_style=:sutton_barto))
    @test env_standard.problem.reward_style == :standard
    @test env_sutton.problem.reward_style == :sutton_barto
end

@testitem "CartPole action and observation spaces" tags = [:environments, :cartpole, :spaces] begin
    using ClassicControlEnvironments

    env = CartPoleEnv()
    
    # Test action space is discrete with 2 actions
    action_space = DRiL.action_space(env)
    @test action_space isa Discrete
    @test action_space.n == 2
    @test action_space.start == 0  # Should be 0-based (0=left, 1=right)
    
    # Test observation space is 4D box
    obs_space = DRiL.observation_space(env)
    @test obs_space isa Box{Float32}
    @test obs_space.shape == (4,)
    
    # Check observation bounds are reasonable
    @test obs_space.low[1] ≈ -4.8f0  # cart position
    @test obs_space.high[1] ≈ 4.8f0
    @test obs_space.low[3] ≈ -0.4189f0  # pole angle (approximately ±24°)
    @test obs_space.high[3] ≈ 0.4189f0
end

@testitem "CartPole basic functionality" tags = [:environments, :cartpole, :basic] begin
    using ClassicControlEnvironments
    using Random

    env = CartPoleEnv()
    
    # Test reset
    DRiL.reset!(env)
    obs = DRiL.observe(env)
    @test length(obs) == 4
    @test obs ∈ DRiL.observation_space(env)
    @test !DRiL.terminated(env)
    @test !DRiL.truncated(env)
    
    # Test initial state is near zero
    @test abs(obs[1]) < 0.1f0  # cart position
    @test abs(obs[2]) < 0.1f0  # cart velocity  
    @test abs(obs[3]) < 0.1f0  # pole angle
    @test abs(obs[4]) < 0.1f0  # pole angular velocity
    
    # Test step with valid action
    action = 1  # Push right
    reward = DRiL.act!(env, action)
    @test reward isa Float32
    @test reward >= 0.0f0  # Standard CartPole gives positive reward per step
    
    # Test step function
    next_obs, step_reward, term, trunc, info = DRiL.step!(env, action)
    @test step_reward == reward
    @test term == DRiL.terminated(env)
    @test trunc == DRiL.truncated(env)
    @test info isa Dict
    @test length(next_obs) == 4
    @test next_obs ∈ DRiL.observation_space(env)
end

@testitem "CartPole action processing" tags = [:environments, :cartpole, :actions] begin
    using ClassicControlEnvironments

    env = CartPoleEnv()
    action_space = DRiL.action_space(env)
    
    # Test that actions are processed correctly
    DRiL.reset!(env)
    
    # Test left action (0)
    reward_left = DRiL.act!(env, 0)
    @test reward_left isa Float32
    
    DRiL.reset!(env)
    
    # Test right action (1)  
    reward_right = DRiL.act!(env, 1)
    @test reward_right isa Float32
    
    # Test with action that needs processing (1-based to 0-based)
    # When policy gives action 1 (1-based), it should be converted to 0 (0-based)
    processed_action_0 = process_action(1, action_space)
    @test processed_action_0 == 0
    
    processed_action_1 = process_action(2, action_space)
    @test processed_action_1 == 1
    
    # Test out-of-bounds clamping
    @test process_action(0, action_space) == 0  # Below range
    @test process_action(3, action_space) == 1  # Above range
end

@testitem "CartPole episode termination" tags = [:environments, :cartpole, :episodes] begin
    using ClassicControlEnvironments

    env = CartPoleEnv()
    
    # Test that pole angle termination works
    DRiL.reset!(env)
    
    # Manually set state to near termination angle
    env.state = Float32[0.0, 0.0, 0.20, 0.0]  # Large pole angle
    
    # Take action - should terminate
    DRiL.act!(env, 0)
    @test DRiL.terminated(env)
    
    # Test cart position termination
    DRiL.reset!(env)
    env.state = Float32[2.3, 0.0, 0.0, 0.0]  # Near cart position limit
    
    DRiL.act!(env, 1)  # Push further right
    @test DRiL.terminated(env)
    
    # Test that normal states don't terminate
    DRiL.reset!(env)
    env.state = Float32[0.0, 0.0, 0.1, 0.0]  # Small angle
    
    DRiL.act!(env, 0)
    @test !DRiL.terminated(env)
end

@testitem "CartPole reward systems" tags = [:environments, :cartpole, :rewards] begin
    using ClassicControlEnvironments

    # Test standard reward (positive per step)
    env_standard = CartPoleEnv(CartPoleProblem(reward_style=:standard))
    DRiL.reset!(env_standard)
    
    reward_standard = DRiL.act!(env_standard, 0)
    @test reward_standard ≈ 1.0f0  # Standard gives +1 per step
    
    # Test that termination still gives reward in standard mode
    env_standard.state = Float32[0.0, 0.0, 0.25, 0.0]  # Terminal angle
    reward_terminal = DRiL.act!(env_standard, 0)
    @test reward_terminal ≈ 1.0f0  # Should still get +1 even on terminal step
    
    # Test Sutton-Barto reward (0 per step, -1 on termination)
    env_sutton = CartPoleEnv(CartPoleProblem(reward_style=:sutton_barto))
    DRiL.reset!(env_sutton)
    
    reward_sutton = DRiL.act!(env_sutton, 0)
    @test reward_sutton ≈ 0.0f0  # Sutton-Barto gives 0 per step
    
    # Test termination penalty in Sutton-Barto mode
    env_sutton.state = Float32[0.0, 0.0, 0.25, 0.0]  # Terminal angle
    reward_sutton_terminal = DRiL.act!(env_sutton, 0)
    @test reward_sutton_terminal ≈ -1.0f0  # Should get -1 on termination
end

@testitem "CartPole physics simulation" tags = [:environments, :cartpole, :physics] begin
    using ClassicControlEnvironments

    env = CartPoleEnv()
    
    # Test that physics behaves reasonably
    DRiL.reset!(env)
    initial_state = copy(env.state)
    
    # Apply consistent rightward force
    for i in 1:10
        DRiL.act!(env, 1)  # Push right
        if DRiL.terminated(env)
            break
        end
    end
    
    # Cart should have moved right
    @test env.state[1] > initial_state[1]  # Cart position increased
    
    # Test leftward force
    DRiL.reset!(env)
    initial_state = copy(env.state)
    
    for i in 1:10
        DRiL.act!(env, 0)  # Push left
        if DRiL.terminated(env)
            break
        end
    end
    
    # Cart should have moved left
    @test env.state[1] < initial_state[1]  # Cart position decreased
    
    # Test that pole angle responds to cart movement
    DRiL.reset!(env)
    env.state = Float32[0.0, 1.0, 0.0, 0.0]  # Cart moving right, pole upright
    
    old_angle = env.state[3]
    DRiL.act!(env, 0)  # Push left (opposing cart motion)
    
    # Pole should tilt due to cart deceleration
    @test abs(env.state[3]) > abs(old_angle)
end

@testitem "CartPole with discrete policy integration" tags = [:environments, :cartpole, :integration] setup = [SharedTestSetup] begin
    using ClassicControlEnvironments
    using Random

    # Test CartPole environment with discrete policy
    env = CartPoleEnv()
    policy = DiscreteActorCriticPolicy(DRiL.observation_space(env), DRiL.action_space(env))
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)
    
    # Test single step interaction
    DRiL.reset!(env)
    obs = DRiL.observe(env)
    
    # Get action from policy
    action, _ = predict(policy, obs, params, states; rng=rng)
    @test action ∈ DRiL.action_space(env)
    
    # Apply action to environment
    reward = DRiL.act!(env, action)
    @test reward isa Float32
    @test reward >= 0.0f0
    
    # Test full episode
    DRiL.reset!(env)
    total_reward = 0.0f0
    steps = 0
    max_steps = 500
    
    while !DRiL.terminated(env) && !DRiL.truncated(env) && steps < max_steps
        obs = DRiL.observe(env)
        action, _ = predict(policy, obs, params, states; rng=rng)
        reward = DRiL.act!(env, action)
        total_reward += reward
        steps += 1
    end
    
    @test steps > 0  # Should take at least one step
    @test total_reward > 0.0f0  # Should accumulate some reward
    @test steps <= max_steps  # Should not exceed max steps
end

@testitem "CartPole parameter variations" tags = [:environments, :cartpole, :parameters] begin
    using ClassicControlEnvironments

    # Test with different gravity
    env_low_gravity = CartPoleEnv(CartPoleProblem(gravity=5.0f0))
    env_high_gravity = CartPoleEnv(CartPoleProblem(gravity=15.0f0))
    
    @test env_low_gravity.problem.gravity == 5.0f0
    @test env_high_gravity.problem.gravity == 15.0f0
    
    # Test with different pole mass
    env_light_pole = CartPoleEnv(CartPoleProblem(mass_pole=0.05f0))
    env_heavy_pole = CartPoleEnv(CartPoleProblem(mass_pole=0.2f0))
    
    @test env_light_pole.problem.mass_pole == 0.05f0
    @test env_heavy_pole.problem.mass_pole == 0.2f0
    @test env_light_pole.problem.total_mass == env_light_pole.problem.mass_cart + 0.05f0
    
    # Test with different thresholds
    env_tight = CartPoleEnv(CartPoleProblem(
        theta_threshold_radians=10.0f0 * π / 180.0f0,  # 10 degrees
        x_threshold=1.5f0
    ))
    
    @test env_tight.problem.theta_threshold_radians ≈ 10.0f0 * π / 180.0f0
    @test env_tight.problem.x_threshold == 1.5f0
    
    # Test that tighter thresholds terminate sooner
    DRiL.reset!(env_tight)
    env_tight.state = Float32[1.4, 0.0, 0.0, 0.0]  # Near position limit
    DRiL.act!(env_tight, 1)
    @test DRiL.terminated(env_tight)  # Should terminate due to tight threshold
end

@testitem "CartPole edge cases" tags = [:environments, :cartpole, :edge_cases] begin
    using ClassicControlEnvironments

    env = CartPoleEnv()
    
    # Test with extreme initial conditions (within reset bounds)
    DRiL.reset!(env)
    
    # Test multiple resets
    for i in 1:10
        DRiL.reset!(env)
        obs = DRiL.observe(env)
        @test obs ∈ DRiL.observation_space(env)
        @test !DRiL.terminated(env)
    end
    
    # Test very short episode (immediate termination)
    DRiL.reset!(env)
    env.state = Float32[2.5, 0.0, 0.0, 0.0]  # Beyond cart threshold
    
    reward = DRiL.act!(env, 0)
    @test DRiL.terminated(env)
    
    # Test with maximum values that shouldn't terminate
    DRiL.reset!(env)
    env.state = Float32[2.3, 0.0, 0.199, 0.0]  # Just below thresholds
    
    reward = DRiL.act!(env, 0)
    @test !DRiL.terminated(env)  # Should not terminate yet
    
    # Test info dictionary contains useful information
    DRiL.reset!(env)
    _, _, _, _, info = DRiL.step!(env, 1)
    @test info isa Dict
    # Info can be empty or contain additional debug information
end 
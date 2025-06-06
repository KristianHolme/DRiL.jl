using Test
using DRiL
using TestItems

@testset "policies.jl" begin
    # TODO: Add tests for policies
end

@testitem "DiscreteActorCriticPolicy construction" tags = [:policies, :discrete, :construction] setup = [SharedTestSetup] begin
    using Random
    using Lux

    # Test basic construction
    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)  # 0-based (like Gymnasium)
    
    policy = DiscreteActorCriticPolicy(obs_space, action_space)
    
    @test policy isa DiscreteActorCriticPolicy
    @test isequal(policy.observation_space, obs_space)
    @test isequal(policy.action_space, action_space)
    @test policy.shared_features == true
    
    # Test with custom parameters
    policy_custom = DiscreteActorCriticPolicy(obs_space, action_space; 
                                             hidden_dim=[32, 16], 
                                             activation=relu, 
                                             shared_features=false)
    @test policy_custom.shared_features == false
    
    # Test with 1-based action space
    action_space_1 = Discrete(3, 1)  # 1-based
    policy_1based = DiscreteActorCriticPolicy(obs_space, action_space_1)
    @test policy_1based.action_space == action_space_1
end

@testitem "DiscreteActorCriticPolicy parameter initialization" tags = [:policies, :discrete, :parameters] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(5, 0)
    policy = DiscreteActorCriticPolicy(obs_space, action_space)
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    
    # Check parameter structure
    @test haskey(params, :feature_extractor)
    @test haskey(params, :actor_head)
    @test haskey(params, :critic_head)
    @test !haskey(params, :log_std)  # Discrete policies shouldn't have log_std
    
    # Check parameter count
    param_count = Lux.parameterlength(policy)
    @test param_count > 0
    @test param_count == length(ComponentVector(params))
    
    # Test state initialization
    states = Lux.initialstates(rng, policy)
    @test haskey(states, :feature_extractor)
    @test haskey(states, :actor_head) 
    @test haskey(states, :critic_head)
end

@testitem "DiscreteActorCriticPolicy prediction" tags = [:policies, :discrete, :prediction] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(3, 0)  # Actions: 0, 1, 2
    policy = DiscreteActorCriticPolicy(obs_space, action_space)
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)
    
    # Test single observation prediction
    obs = Float32[0.5, -0.3]
    actions, new_states = predict(policy, obs, params, states; deterministic=false, rng=rng)
    
    # Actions should be in environment action space after processing
    @test actions ∈ action_space
    @test actions isa Integer
    
    # Test deterministic prediction
    actions_det, _ = predict(policy, obs, params, states; deterministic=true, rng=rng)
    @test actions_det ∈ action_space
    @test actions_det isa Integer
    
    # Test batch prediction
    batch_obs = Float32[0.5 -0.2; -0.3 0.7]  # 2 observations
    batch_actions, _ = predict(policy, batch_obs, params, states; deterministic=false, rng=rng)
    
    @test length(batch_actions) == 2
    @test all(a -> a ∈ action_space, batch_actions)
    @test all(a -> a isa Integer, batch_actions)
end

@testitem "DiscreteActorCriticPolicy action evaluation" tags = [:policies, :discrete, :evaluation] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(4, 0)  # Actions: 0, 1, 2, 3
    policy = DiscreteActorCriticPolicy(obs_space, action_space)
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)
    
    # Test single observation evaluation
    obs = Float32[0.5, -0.3]
    
    # Get action from policy (this will be in 1-based Julia indexing internally)
    actions, values, log_probs, _ = policy(obs, params, states; rng=rng)
    
    # Test that actions are valid indices (1-based for internal use)
    @test actions isa Integer
    @test 1 <= actions <= action_space.n
    
    # Evaluate the same actions
    eval_values, eval_log_probs, entropy, _ = evaluate_actions(policy, obs, actions, params, states)
    
    # Values should match
    @test eval_values ≈ values atol=1e-6
    
    # Log probabilities should match (approximately due to floating point)
    @test eval_log_probs[1] ≈ log_probs atol=1e-5
    
    # Entropy should be positive for stochastic policy
    @test entropy[1] >= 0
    
    # Test batch evaluation
    batch_obs = Float32[0.5 -0.2; -0.3 0.7]
    batch_actions, batch_values, batch_log_probs, _ = policy(batch_obs, params, states; rng=rng)
    
    eval_batch_values, eval_batch_log_probs, batch_entropy, _ = evaluate_actions(policy, batch_obs, batch_actions, params, states)
    
    @test length(eval_batch_values) == 2
    @test length(eval_batch_log_probs) == 2
    @test length(batch_entropy) == 2
    @test eval_batch_values ≈ batch_values atol=1e-6
    @test all(eval_batch_log_probs .≈ batch_log_probs)
end

@testitem "DiscreteActorCriticPolicy indexing consistency" tags = [:policies, :discrete, :indexing] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    
    # Test different action space configurations
    spaces_to_test = [
        Discrete(3, 0),   # 0-based (Gymnasium style): 0, 1, 2
        Discrete(3, 1),   # 1-based (Julia style): 1, 2, 3
        Discrete(4, -1),  # Custom start: -1, 0, 1, 2
    ]
    
    for action_space in spaces_to_test
        policy = DiscreteActorCriticPolicy(obs_space, action_space)
        
        rng = Random.MersenneTwister(42)
        params = Lux.initialparameters(rng, policy)
        states = Lux.initialstates(rng, policy)
        
        obs = Float32[0.5, -0.3]
        
        # Test that policy actions (before processing) are in 1-based indexing
        actions, _, _, _ = policy(obs, params, states; rng=rng)
        @test 1 <= actions <= action_space.n  # Internal actions should be 1-based
        
        # Test that predict() returns processed actions in action space range
        processed_actions, _ = predict(policy, obs, params, states; rng=rng)
        @test processed_actions ∈ action_space  # Should be in action space range
        
        # Test that evaluation works with stored actions (1-based)
        eval_values, eval_log_probs, entropy, _ = evaluate_actions(policy, obs, actions, params, states)
        @test length(eval_log_probs) == 1
        @test length(entropy) == 1
        @test eval_log_probs[1] isa Float32
        @test entropy[1] >= 0
    end
end

@testitem "DiscreteActorCriticPolicy action space conversion" tags = [:policies, :discrete, :conversion] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    action_space = Discrete(3, 0)  # Gymnasium style: 0, 1, 2
    policy = DiscreteActorCriticPolicy(obs_space, action_space)
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)
    
    obs = Float32[0.5, -0.3]
    
    # Test multiple predictions to check conversion consistency
    for i in 1:10
        # Get raw policy action (1-based internally)
        raw_action, _, _, _ = policy(obs, params, states; rng=rng)
        @test 1 <= raw_action <= 3  # Should be in [1, 2, 3]
        
        # Get processed action for environment
        env_action, _ = predict(policy, obs, params, states; rng=rng)
        @test env_action ∈ action_space  # Should be in [0, 1, 2]
        @test 0 <= env_action <= 2
        
        # Test that process_action works correctly
        manual_processed = process_action(raw_action, action_space)
        @test manual_processed ∈ action_space
        @test manual_processed == raw_action + (action_space.start - 1)  # 1-based to 0-based
    end
end

@testitem "DiscreteActorCriticPolicy vs ContinuousActorCriticPolicy interface" tags = [:policies, :discrete, :interface] setup = [SharedTestSetup] begin
    using Random

    obs_space = Box(Float32[-1.0, -1.0], Float32[1.0, 1.0])
    discrete_action_space = Discrete(4, 0)
    continuous_action_space = Box(Float32[-1.0], Float32[1.0])
    
    discrete_policy = DiscreteActorCriticPolicy(obs_space, discrete_action_space)
    continuous_policy = ContinuousActorCriticPolicy(obs_space, continuous_action_space)
    
    # Test that both policies implement the same interface
    rng = Random.MersenneTwister(42)
    
    discrete_params = Lux.initialparameters(rng, discrete_policy)
    discrete_states = Lux.initialstates(rng, discrete_policy)
    
    continuous_params = Lux.initialparameters(rng, continuous_policy)
    continuous_states = Lux.initialstates(rng, continuous_policy)
    
    obs = Float32[0.5, -0.3]
    
    # Test that both implement the same methods
    discrete_actions, discrete_values, discrete_log_probs, _ = discrete_policy(obs, discrete_params, discrete_states; rng=rng)
    continuous_actions, continuous_values, continuous_log_probs, _ = continuous_policy(obs, continuous_params, continuous_states; rng=rng)
    
    # Test predict
    discrete_pred, _ = predict(discrete_policy, obs, discrete_params, discrete_states; rng=rng)
    continuous_pred, _ = predict(continuous_policy, obs, continuous_params, continuous_states; rng=rng)
    
    # Test predict_values
    discrete_vals, _ = predict_values(discrete_policy, obs, discrete_params, discrete_states)
    continuous_vals, _ = predict_values(continuous_policy, obs, continuous_params, continuous_states)
    
    # Test evaluate_actions
    discrete_eval_values, discrete_eval_log_probs, discrete_entropy, _ = evaluate_actions(discrete_policy, obs, discrete_actions, discrete_params, discrete_states)
    continuous_eval_values, continuous_eval_log_probs, continuous_entropy, _ = evaluate_actions(continuous_policy, obs, continuous_actions, continuous_params, continuous_states)
    
    # Test that outputs have expected types and shapes
    @test discrete_actions isa Integer
    @test continuous_actions isa AbstractArray
    @test discrete_pred isa Integer  
    @test continuous_pred isa AbstractArray
    @test discrete_vals isa AbstractArray
    @test continuous_vals isa AbstractArray
    @test length(discrete_eval_log_probs) == 1
    @test length(continuous_eval_log_probs) == 1
end

@testitem "DiscreteActorCriticPolicy edge cases" tags = [:policies, :discrete, :edge_cases] setup = [SharedTestSetup] begin
    using Random

    # Test single action space
    obs_space = Box(Float32[-1.0], Float32[1.0])
    single_action_space = Discrete(1, 0)  # Only action 0
    policy = DiscreteActorCriticPolicy(obs_space, single_action_space)
    
    rng = Random.MersenneTwister(42)
    params = Lux.initialparameters(rng, policy)
    states = Lux.initialstates(rng, policy)
    
    obs = Float32[0.5]
    
    # Test that single action space works
    actions, values, log_probs, _ = policy(obs, params, states; rng=rng)
    @test actions == 1  # Should always be 1 (1-based internally)
    
    processed_action, _ = predict(policy, obs, params, states; rng=rng)
    @test processed_action == 0  # Should be 0 after processing
    
    # Test large action space
    large_action_space = Discrete(100, 0)
    large_policy = DiscreteActorCriticPolicy(obs_space, large_action_space)
    
    large_params = Lux.initialparameters(rng, large_policy)
    large_states = Lux.initialstates(rng, large_policy)
    
    large_actions, _, _, _ = large_policy(obs, large_params, large_states; rng=rng)
    @test 1 <= large_actions <= 100  # Internal action should be in [1, 100]
    
    large_processed, _ = predict(large_policy, obs, large_params, large_states; rng=rng)
    @test 0 <= large_processed <= 99  # Processed should be in [0, 99]
    
    # Test negative start action space
    neg_action_space = Discrete(5, -2)  # Actions: -2, -1, 0, 1, 2
    neg_policy = DiscreteActorCriticPolicy(obs_space, neg_action_space)
    
    neg_params = Lux.initialparameters(rng, neg_policy)
    neg_states = Lux.initialstates(rng, neg_policy)
    
    neg_actions, _, _, _ = neg_policy(obs, neg_params, neg_states; rng=rng)
    @test 1 <= neg_actions <= 5  # Internal should be [1, 5]
    
    neg_processed, _ = predict(neg_policy, obs, neg_params, neg_states; rng=rng)
    @test -2 <= neg_processed <= 2  # Processed should be [-2, 2]
end 
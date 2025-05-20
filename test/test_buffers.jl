using Test
using DRiL
using Pendulum
@testset "buffers.jl" begin
    # Test: logprobs are consistent after rollout
    pend_env() = PendulumEnv() |> ScalingWrapperEnv
    env = MultiThreadedParallelEnv([pend_env() for _ in 1:4])
    policy = ActorCriticPolicy(observation_space(env), action_space(env))
    agent = ActorCriticAgent(policy; n_steps=8, batch_size=8, epochs=1, verbose=0)
    alg = PPO()
    n_steps = agent.n_steps
    n_envs = 1
    roll_buffer = RolloutBuffer(observation_space(env), action_space(env), alg.gae_lambda, alg.gamma, n_steps, n_envs)
    for i in 1:10
        DRiL.collect_rollouts!(roll_buffer, agent, env)
        obs = roll_buffer.observations
        act = roll_buffer.actions
        logprobs = roll_buffer.logprobs
        ps = agent.train_state.parameters
        st = agent.train_state.states
        _, new_logprobs, _, _ = evaluate_actions(policy, obs, act, ps, st)
        @test isapprox(vec(logprobs), vec(new_logprobs); atol=1e-5, rtol=1e-5)
    end
end
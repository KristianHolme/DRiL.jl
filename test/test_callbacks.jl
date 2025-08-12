@testitem "callbacks locals" begin
    using ClassicControlEnvironments
    using Zygote
    ##
    alg = PPO(; ent_coef=0.1f0, n_steps=256, batch_size=64, epochs=10)
    env = BroadcastedParallelEnv([MountainCarContinuousEnv() for _ in 1:8])
    env = MonitorWrapperEnv(env)
    env = NormalizeWrapperEnv(env, gamma=alg.gamma)

    policy = ActorCriticPolicy(observation_space(env), action_space(env))
    agent = ActorCriticAgent(policy, alg; verbose=2)

    function test_keys(locals::Dict, keys_to_check::Vector{Symbol})
        @info "locals keys: $(keys(locals))"
        for key in keys_to_check
            key_in_locals = haskey(locals, key)
            @test key_in_locals
            if !key_in_locals
                @info "key $key not in locals"
            end
        end
        true
    end

    @kwdef struct OnTrainingStartCheckLocalsCallback <: AbstractCallback
        keys::Vector{Symbol} = [:agent, :env, :alg, :iterations, :total_steps, :max_steps,
            :n_steps, :n_envs, :roll_buffer, :iterations, :total_fps, :callbacks]
    end
    function DRiL.on_training_start(callback::OnTrainingStartCheckLocalsCallback, locals::Dict)
        @info "OnTrainingStartCheckLocalsCallback"
        test_keys(locals, callback.keys)
        true
    end

    @kwdef struct OnRolloutStartCheckLocalsCallback <: AbstractCallback
        first_keys::Vector{Symbol} = [:agent, :env, :alg, :iterations, :total_steps,
            :max_steps, :i, :learning_rate]
        subsequent_keys::Vector{Symbol} = [:agent, :env, :alg, :iterations, :total_steps, :max_steps]
    end
    function DRiL.on_rollout_start(callback::OnRolloutStartCheckLocalsCallback, locals::Dict)
        @info "OnRolloutStartCheckLocalsCallback"
        test_keys(locals, callback.first_keys)
        if locals[:i] > 1
            test_keys(locals, callback.subsequent_keys)
        end
        true
    end
    learn!(agent, env, alg, 3000; callbacks=[
        OnTrainingStartCheckLocalsCallback(),
        OnRolloutStartCheckLocalsCallback()
    ]
    )

end
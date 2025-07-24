@kwdef struct SAC{T<:AbstractFloat,E<:AbstractEntropyCoefficient} <: AbstractAlgorithm
    learning_rate::T = 3f-4
    buffer_size::Int = 1_000_000
    start_steps::Int = 100 # how many steps to collect before first gradient update
    batch_size::Int = 256
    tau::T = 0.005 #soft update rate
    gamma::T = 0.99 #discount
    train_freq::Int = 1
    gradient_steps::Int = 1 # -1 to do as many updates as steps (train_freq)
    ent_coef::E = AutoEntropyCoefficient()
    target_update_interval::Int = 1 # how often to update the target networks
end

function sac_ent_coef_loss(alg::SAC{T,AutoEntropyCoefficient{T,FixedEntropyTarget{T}}},
    policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, ps, st, data
) where {T}
    log_ent_coef = ps.log_ent_coef[1]
    policy_ps = data.policy_ps
    policy_st = data.policy_st
    _, log_probs_pi, policy_st = action_log_prob(policy, data.obs, policy_ps, policy_st)
    target_entropy = alg.ent_coef.target.target
    loss = -(log_ent_coef * @ignore_derivatives(log_probs_pi .+ target_entropy |> mean))
    return loss, st, Dict("policy_st" => policy_st)
end

function sac_actor_loss(alg::SAC, policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, actor_ps, actor_st, data)
    obs = data.obs
    ent_coef = data.log_ent_coef[1] |> exp
    critic_ps = data.critic_ps
    critic_st = data.critic_st
    actions_pi, log_probs_pi, actor_st = action_log_prob(policy, obs, ps, st)
    q_values, st = @ignore_derivatives predict_values(policy, obs, actions_pi, critic_ps, critic_st)
    min_q_values = minimum(q_values, dims=1) |> vec
    loss = mean(ent_coef .* log_probs_pi - min_q_values)
    return loss, st, Dict()
end

function sac_critic_loss(alg::SAC, policy::ContinuousActorCriticPolicy{<:Any,<:Any,<:Any,QCritic}, ps, st, data)
    obs, actions, rewards, next_obs = data.observations, data.actions, data.rewards, data.next_observations
    dones = data.terminated .|| data.truncated
    gamma = alg.gamma
    ent_coef = data.log_ent_coef[1] |> exp
    target_ps = data.target_ps
    target_st = data.target_st

    # Current Q-values
    current_q_values, new_st = predict_values(policy, obs, actions, ps, st)

    # Target Q-values (no gradients)
    target_q_values = @ignore_derivatives begin
        next_actions, next_log_probs, st = action_log_prob(policy, next_obs, ps, st)
        #replace critic ps and st with target
        ps_with_target = merge(ps, target_ps)
        st_with_target = merge(st, target_st)
        next_q_vals, _ = predict_values(policy, next_obs, next_actions, ps_with_target, st_with_target)
        min_next_q = minimum(next_q_vals, dims=1) |> vec

        # Add entropy term
        next_q_vals_with_entropy = min_next_q .- ent_coef .* next_log_probs

        # Bellman target
        rewards .+ (1 .- dones) .* gamma .* next_q_vals_with_entropy
    end

    # Critic loss (sum over all Q-networks)
    critic_loss = sum(mean((current_q .- target_q_values) .^ 2) for current_q in eachrow(current_q_values))

    return critic_loss, new_st, Dict()
end
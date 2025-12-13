# RolloutBuffer-specific implementations

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs

#TODO:fix types here
function RolloutBuffer(observation_space::AbstractSpace, action_space::AbstractSpace, gae_lambda::T, gamma::T, n_steps::Int, n_envs::Int) where {T <: AbstractFloat}
    total_steps = n_steps * n_envs
    obs_eltype = eltype(observation_space)
    action_eltype = eltype(action_space)
    observations = Array{obs_eltype}(undef, size(observation_space)..., total_steps)
    actions = Array{action_eltype}(undef, size(action_space)..., total_steps)
    rewards = Vector{T}(undef, total_steps)
    advantages = Vector{T}(undef, total_steps)
    returns = Vector{T}(undef, total_steps)
    logprobs = Vector{T}(undef, total_steps)
    values = Vector{T}(undef, total_steps)
    return RolloutBuffer{T, obs_eltype, action_eltype}(observations, actions, rewards, advantages, returns, logprobs, values, gae_lambda, gamma, n_steps, n_envs)
end

"""
    collect!(rollout_buffer, agent, alg::OnPolicyAlgorithm, env; kwargs...)

Thin wrapper over collect_rollout! for on-policy algorithms.
"""
function collect!(
        rollout_buffer::RolloutBuffer,
        agent::Agent,
        alg::OnPolicyAlgorithm,
        env::AbstractEnv;
        kwargs...
    )
    return collect_rollout!(rollout_buffer, agent, alg, env; kwargs...)
end

function reset!(rollout_buffer::RolloutBuffer)
    rollout_buffer.observations .= 0
    rollout_buffer.actions .= 0
    rollout_buffer.rewards .= 0
    rollout_buffer.advantages .= 0
    rollout_buffer.returns .= 0
    rollout_buffer.logprobs .= 0
    rollout_buffer.values .= 0
    return nothing
end

function collect_rollout!(
        rollout_buffer::RolloutBuffer,
        agent::Agent,
        alg::OnPolicyAlgorithm,
        env::AbstractEnv;
        callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing
    )
    # reset!(env) #we dont reset the, we continue from where we left off

    obs_space = observation_space(env)
    act_space = action_space(env)

    reset!(rollout_buffer)

    t_start = time()
    trajectories, success = collect_trajectories(agent, env, alg, rollout_buffer.n_steps; callbacks = callbacks)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / t_collect
    if !success
        @warn "Collecting trajectories stopped due to callback failure"
        return fps, false
    end

    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer
        traj_inds = positions[i]:(positions[i + 1] - 1)
        # @debug "traj_inds: $(traj_inds)"
        selectdim(rollout_buffer.observations, ndims(obs_space) + 1, traj_inds) .= batch(traj.observations, obs_space)
        selectdim(rollout_buffer.actions, ndims(act_space) + 1, traj_inds) .= batch(traj.actions, act_space)
        rollout_buffer.rewards[traj_inds] .= traj.rewards
        rollout_buffer.logprobs[traj_inds] .= traj.logprobs
        rollout_buffer.values[traj_inds] .= traj.values

        #compute advantages and returns
        compute_advantages!(
            @view(rollout_buffer.advantages[traj_inds]),
            traj, rollout_buffer.gamma, rollout_buffer.gae_lambda
        )
        rollout_buffer.returns[traj_inds] .= rollout_buffer.advantages[traj_inds] .+ rollout_buffer.values[traj_inds]
    end
    return fps, true
end

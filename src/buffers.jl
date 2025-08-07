struct RolloutBuffer{T<:AbstractFloat,O,A} <: AbstractBuffer
    observations::Array{O}
    actions::Array{A}
    rewards::Vector{T}
    advantages::Vector{T}
    returns::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    gae_lambda::T
    gamma::T
    n_steps::Int
    n_envs::Int
end

Base.length(rb::RolloutBuffer) = rb.n_steps * rb.n_envs

#TODO:fix types here
function RolloutBuffer(observation_space::AbstractSpace, action_space::AbstractSpace, gae_lambda::T, gamma::T, n_steps::Int, n_envs::Int) where {T<:AbstractFloat}
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
    return RolloutBuffer{T,obs_eltype,action_eltype}(observations, actions, rewards, advantages, returns, logprobs, values, gae_lambda, gamma, n_steps, n_envs)
end


function reset!(rollout_buffer::RolloutBuffer)
    rollout_buffer.observations .= 0
    rollout_buffer.actions .= 0
    rollout_buffer.rewards .= 0
    rollout_buffer.advantages .= 0
    rollout_buffer.returns .= 0
    rollout_buffer.logprobs .= 0
    rollout_buffer.values .= 0
    nothing
end

mutable struct Trajectory{T<:AbstractFloat,O,A}
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{T}
    logprobs::Vector{T}
    values::Vector{T}
    terminated::Bool
    truncated::Bool
    bootstrap_value::Union{Nothing,T}  # Value of the next state for truncated episodes
end
function Trajectory{T}(observation_space::AbstractSpace, action_space::AbstractSpace) where {T<:AbstractFloat}
    obs_type = typeof(rand(observation_space))
    action_type = typeof(rand(action_space))
    observations = Array{obs_type}[]
    actions = Array{action_type}[]
    rewards = T[]
    logprobs = T[]
    values = T[]
    terminated = false
    truncated = false
    bootstrap_value = nothing
    return Trajectory{T,obs_type,action_type}(observations, actions, rewards, logprobs, values, terminated, truncated, bootstrap_value)
end

Trajectory(observation_space::AbstractSpace, action_space::AbstractSpace) = Trajectory{Float32}(observation_space, action_space)

Base.length(trajectory::Trajectory) = length(trajectory.rewards)
total_reward(trajectory::Trajectory) = sum(trajectory.rewards)


function collect_trajectories(agent::ActorCriticAgent, env::AbstractParallelEnv, n_steps::Int,
    progress_meter::Union{Progress,Nothing}=nothing; callbacks::Union{Vector{<:AbstractCallback},Nothing}=nothing)
    # reset!(env)
    trajectories = Trajectory[]
    obs_space = observation_space(env)
    act_space = action_space(env)
    n_envs = number_of_envs(env)
    current_trajectories = [Trajectory(obs_space, act_space) for _ in 1:n_envs]
    new_obs = observe(env)
    for i in 1:n_steps
        all_good = true
        if !isnothing(callbacks)
            for callback in callbacks
                callback_good = on_step(callback, agent, env, current_trajectories, new_obs)
                all_good = all_good && callback_good
            end
            if !all_good
                @warn "Collecting trajectories stopped due to callback failure"
                return trajectories
            end
        end
        observations = new_obs
        actions, values, logprobs = get_action_and_values(agent, observations)
        processed_actions = process_action.(actions, Ref(act_space))
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        new_obs = observe(env)
        for j in 1:n_envs
            push!(current_trajectories[j].observations, observations[j])
            push!(current_trajectories[j].actions, actions[j])
            push!(current_trajectories[j].rewards, rewards[j])
            push!(current_trajectories[j].logprobs, logprobs[j])
            push!(current_trajectories[j].values, values[j])
            if terminateds[j] || truncateds[j] || i == n_steps
                current_trajectories[j].terminated = terminateds[j]
                current_trajectories[j].truncated = truncateds[j]

                # Handle bootstrapping for truncated episodes
                if truncateds[j] && haskey(infos[j], "terminal_observation")
                    last_observation = infos[j]["terminal_observation"]
                    terminal_value = predict_values(agent, [last_observation])[1]
                    current_trajectories[j].bootstrap_value = terminal_value
                end

                # Handle bootstrapping for rollout-limited trajectories (neither terminated nor truncated)
                # We need to bootstrap with the value of the current observation
                if !terminateds[j] && !truncateds[j] && i == n_steps
                    # Get the next observation after last step (which is the current state)
                    next_obs = new_obs[j]
                    next_value = predict_values(agent, [next_obs])[1]
                    current_trajectories[j].bootstrap_value = next_value
                end

                push!(trajectories, current_trajectories[j])
                current_trajectories[j] = Trajectory(obs_space, act_space)
            end
        end
        !isnothing(progress_meter) && next!(progress_meter, step=number_of_envs(env))
    end
    return trajectories
end

function collect_rollout!(rollout_buffer::RolloutBuffer, agent::ActorCriticAgent, env::AbstractEnv, progress_meter::Union{Progress,Nothing}=nothing; callbacks::Union{Vector{<:AbstractCallback},Nothing}=nothing)
    # reset!(env) #we dont reset the, we continue from where we left off

    obs_space = observation_space(env)
    act_space = action_space(env)

    reset!(rollout_buffer)

    t_start = time()
    trajectories = collect_trajectories(agent, env, rollout_buffer.n_steps, progress_meter; callbacks=callbacks)
    t_collect = time() - t_start
    total_steps = sum(length.(trajectories))
    fps = total_steps / t_collect

    traj_lengths = length.(trajectories)
    positions = cumsum([1; traj_lengths])
    for (i, traj) in enumerate(trajectories)
        #transfer data to the Rolloutbuffer 
        traj_inds = positions[i]:positions[i+1]-1
        # @debug "traj_inds: $(traj_inds)"
        selectdim(rollout_buffer.observations, ndims(obs_space) + 1, traj_inds) .= batch(traj.observations, obs_space)
        selectdim(rollout_buffer.actions, ndims(act_space) + 1, traj_inds) .= batch(traj.actions, act_space)
        rollout_buffer.rewards[traj_inds] .= traj.rewards
        rollout_buffer.logprobs[traj_inds] .= traj.logprobs
        rollout_buffer.values[traj_inds] .= traj.values

        #compute advantages and returns
        compute_advantages!(@view(rollout_buffer.advantages[traj_inds]),
            traj, rollout_buffer.gamma, rollout_buffer.gae_lambda)
        rollout_buffer.returns[traj_inds] .= rollout_buffer.advantages[traj_inds] .+ rollout_buffer.values[traj_inds]
    end
    return fps
end

function compute_advantages!(advantages::AbstractArray, traj::Trajectory, gamma::AbstractFloat, gae_lambda::AbstractFloat)
    n = length(traj.rewards)

    # For terminated episodes, no bootstrapping (bootstrap_value should be nothing)
    # For truncated episodes or rollout-limited trajectories, bootstrap with the next state value
    if traj.terminated || isnothing(traj.bootstrap_value)
        # No bootstrapping for terminated episodes
        delta = traj.rewards[end] - traj.values[end]
    else
        # Bootstrap for truncated or rollout-limited trajectories
        delta = traj.rewards[end] + gamma * traj.bootstrap_value - traj.values[end]
    end

    advantages[end] = delta

    # Compute advantages for earlier steps using the standard GAE recursion
    for i in (n-1):-1:1
        delta = traj.rewards[i] + gamma * traj.values[i+1] - traj.values[i]
        advantages[i] = delta + gamma * gae_lambda * advantages[i+1]
    end

    nothing
end


"""
    OffPolicyTrajectory{T,O,A}

A mutable container for storing a single trajectory of off-policy experience data including observations, actions, rewards, and termination information.
"""
mutable struct OffPolicyTrajectory{T<:AbstractFloat,O,A}
    observations::Vector{O}
    actions::Vector{A}
    rewards::Vector{T}
    terminated::Bool
    truncated::Bool
    truncated_observation::Union{Nothing,O}
end

function OffPolicyTrajectory(observation_space::AbstractSpace, action_space::AbstractSpace)
    obs_type = typeof(rand(observation_space))
    action_type = typeof(rand(action_space))
    obs_scalar_type = eltype(observation_space)
    action_scalar_type = eltype(action_space)
    @assert obs_scalar_type == action_scalar_type "Observation and action types must be the same"
    T = obs_scalar_type
    return OffPolicyTrajectory{T,obs_type,action_type}(obs_type[], action_type[], T[], false, false, nothing)
end

Base.length(traj::OffPolicyTrajectory) = length(traj.rewards)




"""
    ReplayBuffer{T,O,A}

A circular buffer for storing multiple trajectories of off-policy experience data, used for replay-based learning algorithms.

# Truncation Logic
- If `terminated = true`, then there should be no `truncated_observation`
- If `truncated = true`, then there should be a `truncated_observation`  
- If `terminated = false` and `truncated = false`, then we stopped in the middle of an episode, so there should be a `truncated_observation`
"""
struct ReplayBuffer{T,O,A}
    observation_space::AbstractSpace
    action_space::Box
    observations::CircularBuffer{O}
    actions::CircularBuffer{A}
    rewards::CircularBuffer{T}
    terminated::CircularBuffer{Bool}
    truncated::CircularBuffer{Bool}
    truncated_observations::CircularBuffer{Union{Nothing,O}}
end
function ReplayBuffer(observation_space::AbstractSpace, action_space::Box, capacity::Int)
    O = typeof(rand(observation_space))
    A = typeof(rand(action_space))
    obs_scalar_type = eltype(observation_space)
    action_scalar_type = eltype(action_space)
    @assert obs_scalar_type == action_scalar_type "Observation and action types must be the same"
    T = obs_scalar_type
    return ReplayBuffer{T,O,A}(
        observation_space,
        action_space,
        CircularBuffer{O}(capacity),
        CircularBuffer{A}(capacity),
        CircularBuffer{T}(capacity),
        CircularBuffer{Bool}(capacity),
        CircularBuffer{Bool}(capacity),
        CircularBuffer{Union{Nothing,O}}(capacity)
    )
end

observation_space(buffer::ReplayBuffer) = buffer.observation_space
action_space(buffer::ReplayBuffer) = buffer.action_space

function Base.length(buffer::ReplayBuffer)
    obs_len = length(buffer.observations)
    action_len = length(buffer.actions)
    reward_len = length(buffer.rewards)
    terminated_len = length(buffer.terminated)
    truncated_len = length(buffer.truncated)
    truncated_obs_len = length(buffer.truncated_observations)
    @assert allequal([obs_len, action_len, reward_len, terminated_len, truncated_len, truncated_obs_len]) "All buffers must have the same length"
    return obs_len
end
Base.size(buffer::ReplayBuffer) = length(buffer)

function DataStructures.isfull(buffer::ReplayBuffer)
    obs_full = isfull(buffer.observations)
    action_full = isfull(buffer.actions)
    reward_full = isfull(buffer.rewards)
    terminated_full = isfull(buffer.terminated)
    truncated_full = isfull(buffer.truncated)
    truncated_obs_full = isfull(buffer.truncated_observations)
    @assert allequal([obs_full, action_full, reward_full, terminated_full, truncated_full, truncated_obs_full]) "All buffers must have the same length"
    return obs_full
end

function Base.empty!(buffer::ReplayBuffer)
    empty!(buffer.observations)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.terminated)
    empty!(buffer.truncated)
    empty!(buffer.truncated_observations)
    nothing
end

function DataStructures.capacity(buffer::ReplayBuffer)
    obs_cap = capacity(buffer.observations)
    action_cap = capacity(buffer.actions)
    reward_cap = capacity(buffer.rewards)
    terminated_cap = capacity(buffer.terminated)
    truncated_cap = capacity(buffer.truncated)
    truncated_obs_cap = capacity(buffer.truncated_observations)
    @assert allequal([obs_cap, action_cap, reward_cap, terminated_cap, truncated_cap, truncated_obs_cap]) "All buffers must have the same capacity"
    return obs_cap
end

#TODO: make tests
function Base.push!(buffer::ReplayBuffer, traj::OffPolicyTrajectory)
    push!(buffer.observations, traj.observations...)
    push!(buffer.actions, traj.actions...)
    push!(buffer.rewards, traj.rewards...)
    vec_terminated = fill(false, length(traj.observations))
    vec_terminated[end] = traj.terminated
    push!(buffer.terminated, vec_terminated...)
    vec_truncated = fill(false, length(traj.observations))
    vec_truncated[end] = traj.truncated
    push!(buffer.truncated, vec_truncated...)
    O = typeof(traj.observations[1])
    vec_truncated_obs = Vector{Union{Nothing,O}}(nothing, length(traj.observations))
    vec_truncated_obs[end] = traj.truncated_observation
    push!(buffer.truncated_observations, vec_truncated_obs...)
    nothing
end

function get_data_loader(buffer::ReplayBuffer{T,O,A}, batch_size::Int, batches::Int, shuffle::Bool, parallel::Bool, rng::AbstractRNG) where {T,O,A}
    buffer_size = length(buffer)
    samples = batch_size * batches
    sample_inds = sample(rng, 1:buffer_size, samples, replace=true)

    obs_sample = batch(buffer.observations[sample_inds], observation_space(buffer))
    action_sample = batch(buffer.actions[sample_inds], action_space(buffer))
    reward_sample = buffer.rewards[sample_inds]
    terminated_sample = buffer.terminated[sample_inds]
    truncated_sample = buffer.truncated[sample_inds]
    truncated_obs_sample = buffer.truncated_observations[sample_inds]

    next_obs_sample = Vector{O}(undef, count(!, terminated_sample))
    next_obs_ind = 1
    for i in 1:samples
        if !terminated_sample[i]
            next_obs = if isnothing(truncated_obs_sample[i])
                #step is in the middle of a rollout, so we just take the next observation
                buffer.observations[sample_inds[i]+1]
            else
                #step is at end of rollout, or truncated by episode time limit
                truncated_obs_sample[i]
            end
            next_obs_sample[next_obs_ind] = next_obs
            next_obs_ind += 1
        end
    end
    next_obs_sample = batch(next_obs_sample, observation_space(buffer))
    #check that all elements are assigned
    @assert all(x -> isassigned(next_obs_sample, x), eachindex(next_obs_sample))

    return DataLoader((observations=obs_sample, actions=action_sample,
            rewards=reward_sample, terminated=terminated_sample,
            truncated=truncated_sample, next_observations=next_obs_sample);
        batchsize=batch_size, shuffle, parallel, rng)
end


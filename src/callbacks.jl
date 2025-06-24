function on_training_start(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, alg::AbstractAlgorithm, iterations::Int, total_steps::Int)
    return true
end
function on_rollout_start(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, rollout_buffer::AbstractBuffer)
    return true
end
function on_step(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, trajectories::Vector{<:Trajectory}, observations::Vector)
    return true
end
function on_rollout_end(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, rollout_buffer::AbstractBuffer)
    return true
end
function on_training_end(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, alg::AbstractAlgorithm)
    return true
end

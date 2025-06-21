on_training_start(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, alg::AbstractAlgorithm, iterations::Int, total_steps::Int) = true
on_rollout_start(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, rollout_buffer::AbstractBuffer) = true
on_step(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, trajectories::Vector{DRiL.Trajectory}, observations::Vector{Any}) = true
on_rollout_end(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, rollout_buffer::AbstractBuffer) = true
on_training_end(callback::AbstractCallback, agent::AbstractAgent, env::AbstractParallelEnv, alg::AbstractAlgorithm) = true

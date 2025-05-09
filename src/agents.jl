function get_action_and_values(agent::AbstractAgent, observations::AbstractArray)
    return _call(agent.policy, observations, agent.ps, agent.st)
end
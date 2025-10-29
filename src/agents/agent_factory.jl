# Unified agent constructor

"""
    Agent(model, alg; kwargs...)

Create an agent given a model (training-time actor-critic) and an algorithm.
Dispatches to on- or off-policy agent constructors.
"""
function Agent(model, alg::OnPolicyAlgorithm; kwargs...)
    return ActorCriticAgent(model, alg; kwargs...)
end

function Agent(model, alg::OffPolicyAlgorithm; kwargs...)
    # For now we assume continuous control for off-policy support in DRiL
    return OffPolicyActorCriticAgent(model, alg; kwargs...)
end

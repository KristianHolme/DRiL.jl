# Unified agent constructor

"""
    Agent(model, alg; kwargs...)

Create an agent given a model (training-time actor-critic) and an algorithm.
Per-algorithm constructors must be defined as `Agent(model, ::YourAlg; ...)`.
"""
function Agent(model, alg::OnPolicyAlgorithm; kwargs...)
    error("No Agent constructor found for $(typeof(alg)). Define `Agent(model, ::$(typeof(alg)); ...)` in the algorithm file.")
end

function Agent(model, alg::OffPolicyAlgorithm; kwargs...)
    error("No Agent constructor found for $(typeof(alg)). Define `Agent(model, ::$(typeof(alg)); ...)` in the algorithm file.")
end

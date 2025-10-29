# Algorithm traits and adapter selection

"""
    action_adapter(alg) -> AbstractActionAdapter

Return the action adapter to convert policy-space actions to env-space actions
for the given algorithm. Algorithms should extend this.
"""
function action_adapter(alg)
    error("action_adapter not implemented for $(typeof(alg))")
end

# Capability traits (defaults)
has_twin_critics(::AbstractAlgorithm) = false
has_target_networks(::AbstractAlgorithm) = false
has_entropy_tuning(::AbstractAlgorithm) = false

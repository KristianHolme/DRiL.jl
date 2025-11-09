abstract type AbstractPolicy end

"""
    (::AbstractPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())

Return env-space actions for a single observation or a vector of observations.
"""
function (::AbstractPolicy) end

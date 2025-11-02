abstract type AbstractPolicy end

"""
    (::AbstractPolicy)(obs_batch;deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())

Return env-space actions for a single observation or a batch of observations.
"""
function (::AbstractPolicy) end

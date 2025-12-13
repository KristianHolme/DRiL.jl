"""
    AbstractPolicy

Abstract type for deployment policies. Subtypes are callable with signature:

    (policy::AbstractPolicy)(obs; deterministic::Bool = true, rng::AbstractRNG = Random.default_rng())

Return env-space actions for a single observation or a vector of observations.
"""
abstract type AbstractPolicy end

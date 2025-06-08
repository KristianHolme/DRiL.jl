module DRiLDistributions

using Random
using ChainRulesCore

export Categorical, DiagGaussian

export logpdf, entropy, mode


abstract type AbstractDistribution end

abstract type AbstractContinuousDistribution <: AbstractDistribution end

abstract type AbstractDiscreteDistribution <: AbstractDistribution end

include("categorical.jl")
include("diagGaussian.jl")

end
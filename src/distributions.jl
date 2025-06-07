"""
custom implementations of distributions. 

functions:
extend Random.rand
logpdf to get logprobs
entropy to get entropy

distributions:
Categorical

DiagGaussian
also used for action spaces with only one action element
    - mean is the action mean, doesnt need to be a vector, to preserve action shape
    - std, same shape as mean
"""

abstract type AbstractDistribution end

abstract type AbstractContinuousDistribution <: AbstractDistribution end

abstract type AbstractDiscreteDistribution <: AbstractDistribution end

struct Categorical <: AbstractDiscreteDistribution
    probabilities::AbstractVector{<:Real}
    
    # Inner constructor to ensure probabilities sum to 1
    function Categorical(probabilities::AbstractVector{<:Real})
        @assert all(probabilities .>= 0) "All probabilities must be non-negative"
        @assert sum(probabilities) > 0 "Sum of probabilities must be positive"
        
        # Normalize probabilities
        normalized_probs = probabilities ./ sum(probabilities)
        new(normalized_probs)
    end
end

# Constructor from logits (more numerically stable)
function Categorical(; logits::AbstractVector{<:Real})
    # Use log-sum-exp trick for numerical stability
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)
    return Categorical(probabilities)
end

struct DiagGaussian <: AbstractContinuousDistribution
    mean::AbstractArray{<:Real}
    std::AbstractArray{<:Real}
end
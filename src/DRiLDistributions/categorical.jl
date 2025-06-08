"""
custom implementations of distributions. 

functions:
extend Random.rand
logpdf to get logprobs
entropy to get entropy
mode to get the mode

distributions:
Categorical

DiagGaussian
also used for action spaces with only one action element
    - mean is the action mean, doesnt need to be a vector, to preserve action shape
    - std, same shape as mean
"""



struct Categorical{V<:AbstractVector{<:Real}} <: AbstractDiscreteDistribution
    probabilities::V
    
    # Inner constructor to ensure probabilities sum to 1
    function Categorical(probabilities::V) where V <: AbstractVector{<:Real}
        @assert all(probabilities .>= 0) "All probabilities must be non-negative"
        @assert sum(probabilities) > 0 "Sum of probabilities must be positive"
        
        # Normalize probabilities
        normalized_probs = probabilities ./ sum(probabilities)
        new{V}(normalized_probs)
    end
end

# Constructor from logits (more numerically stable)
function Categorical(; logits::V) where V <: AbstractVector{<:Real}
    # Use log-sum-exp trick for numerical stability
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)
    return Categorical(probabilities)
end


function logpdf(d::Categorical, x::Integer)
    return log(d.probabilities[x])
end

function entropy(d::Categorical)
    return -sum(d.probabilities .* log.(d.probabilities))
end

function mode(d::Categorical)
    return argmax(d.probabilities)
end


function Random.rand(rng::AbstractRNG, d::Categorical)
    cumulative_probs = cumsum(d.probabilities)
    u = rand(rng)
    idx = findfirst(cumulative_probs .>= u)
    return idx
end

Random.rand(rng::AbstractRNG, d::Categorical, n::Integer) = [rand(rng, d) for _ in 1:n]






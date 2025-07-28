struct SquashedDiagGaussian <: AbstractContinuousDistribution
    DiagGaussian
    function SquashedDiagGaussian(mean::M, log_std::S) where {M<:AbstractArray,S<:AbstractArray}
        @assert size(mean) == size(log_std) "Mean and log_std must have the same shape"
        return new(DiagGaussian(mean, log_std))
    end
end

function Random.rand(rng::AbstractRNG, d::SquashedDiagGaussian)
    sample = rand(rng, d.DiagGaussian)
    return tanh.(sample)
end

function Random.rand(rng::AbstractRNG, d::SquashedDiagGaussian, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

function logpdf(d::SquashedDiagGaussian, x::AbstractArray)
    gaussian_logpdf = logpdf(d.DiagGaussian, x)
    # More numerically stable formula: 2*(log(2) - x - softplus(-2*x)) instead of log(1 - tanh(x)^2)
    # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    #TODO: type stability, getting Float64
    #TODO: make test for this
    correction = 2 * (log(2) .- x .- Lux.softplus.(-2 .* x))
    squashed_logpdf = gaussian_logpdf .- sum(correction)
    return squashed_logpdf
end

#not implemented: entropy, as its implemented directly in the loss

function mode(d::SquashedDiagGaussian)
    return tanh.(d.DiagGaussian.mean)
end
struct DiagGaussian{T<:Real, M<:AbstractArray{T}, S<:AbstractArray{T}} <: AbstractContinuousDistribution
    mean::M
    log_std::S
    function DiagGaussian(mean::M, log_std::S) where {T<:Real, M <: AbstractArray{T}, S <: AbstractArray{T}}
        @assert size(mean) == size(log_std) "Mean and log_std must have the same shape"
        return new{T, M, S}(mean, log_std)
    end
end

function Random.rand(rng::AbstractRNG, d::DiagGaussian{T, M, S}) where {T, M, S}
    # Ensure the random values are of type T for type stability
    #TODO: maybe use ignore_derivatives
    noise = randn(rng, T, size(d.mean))
    return d.mean .+ exp.(d.log_std) .* noise
end

function Random.rand(rng::AbstractRNG, d::DiagGaussian, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

const log2π = log(2π)

function logpdf(d::DiagGaussian{T, M, S}, x::AbstractArray{T}) where {T, M, S}
    k = length(d.mean)
    # Ensure all intermediate calculations maintain precision and return type T
    log_std_sum = sum(d.log_std)
    # Use mapreduce to fuse operations and avoid intermediate arrays
    diff_squared_sum = mapreduce(+, eachindex(x)) do i
        diff = x[i] - d.mean[i]
        var_inv = exp(-2 * d.log_std[i])
        diff^2 * var_inv
    end
    result = -T(0.5) * (2 * log_std_sum + diff_squared_sum + k * T(log2π))
    return result
end

# function logpdf(d::DiagGaussian{T, M, S}, x::AbstractArray{T}) where {T, M, S}
#     @assert size(x) == size(d.mean) "x and d.mean have different shapes"
#     k = length(d.mean)
#     log_std_sum = sum(d.log_std)
#     var_inv = exp.(-2 * d.log_std)
#     diff_squared_sum = sum((x - d.mean) .^ 2 .* var_inv)
#     result = -T(0.5) * (2 * log_std_sum + diff_squared_sum + k * T(log2π))
#     return result
# end

function entropy(d::DiagGaussian{T, M, S}) where {T, M, S}
    k = length(d.mean)
    log_std_sum = sum(d.log_std)
    result = T(0.5) * k * T(1 + log2π) + log_std_sum
    return result
end

function mode(d::DiagGaussian)
    return d.mean
end
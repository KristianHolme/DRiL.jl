abstract type AbstractSpace end
abstract type AbstractBox <: AbstractSpace end

struct UniformBox{T<:Number} <: AbstractBox
    low::T
    high::T
    shape::Tuple{Int}
end

Base.ndims(space::UniformBox) = length(space.shape)
Base.eltype(::UniformBox{T}) where T = T

# Extend Random.rand for UniformBox spaces
"""
    rand([rng], space::UniformBox{T})

Sample a random value from the uniform box space.

# Examples
```julia
space = UniformBox{Float32}(-1.0f0, 1.0f0, (2, 3))
sample = rand(space)  # Returns a 2×3 Float32 array with values in [-1, 1]
```
"""
function Random.rand(rng::AbstractRNG, space::UniformBox{T}) where T
    # Generate random values in [0, 1] with correct type and shape
    unit_random = rand(rng, T, space.shape...)
    # Scale to [low, high] range
    return unit_random .* (space.high - space.low) .+ space.low
end

# Default RNG version
Random.rand(space::UniformBox) = rand(Random.default_rng(), space)

# Multiple samples version
"""
    rand([rng], space::UniformBox{T}, n::Integer)

Sample `n` random values from the uniform box space.

Returns an array where the last dimension has size `n`.
"""
function Random.rand(rng::AbstractRNG, space::UniformBox{T}, n::Integer) where T
    # Generate random values with an extra dimension for n samples
    unit_random = rand(rng, T, space.shape..., n)
    # Scale to [low, high] range
    return unit_random .* (space.high - space.low) .+ space.low
end

Random.rand(space::UniformBox, n::Integer) = rand(Random.default_rng(), space, n)

"""
    sample in space::UniformBox{T}

Check if a sample is within the bounds of the uniform box space.

# Examples
```julia
space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
[0.5f0, -0.3f0] in space  # Returns true
[1.5f0, 0.0f0] in space   # Returns false

# Can also use ∈ symbol
@test action ∈ action_space
```
"""
function Base.in(sample, space::UniformBox{T}) where T
    if !isa(sample, AbstractArray)
        return false
    end

    # Check shape compatibility (allowing for batch dimensions)
    sample_shape = size(sample)
    if length(sample_shape) < length(space.shape)
        return false
    end

    # Check if the leading dimensions match the space shape
    if sample_shape[1:length(space.shape)] != space.shape
        return false
    end

    # Check type compatibility - require exact type match for strict type safety
    if eltype(sample) != T
        return false
    end

    # Check bounds
    return all(space.low .<= sample .<= space.high)
end


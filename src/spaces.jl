abstract type AbstractSpace end
abstract type AbstractBox <: AbstractSpace end

struct UniformBox{T<:Number} <: AbstractBox
    low::T
    high::T
    shape::Tuple{Int}
end

struct Box{T<:Number} <: AbstractBox
    low::Array{T}
    high::Array{T}
    shape::Tuple{Vararg{Int}}
end
function Box{T}(low::Array{T}, high::Array{T}) where T<:Number
    @assert size(low) == size(high) "Low and high arrays must have the same shape"
    @assert all(low .<= high) "All low values must be <= corresponding high values"
    shape = size(low)
    return Box{T}(low, high, shape)
end
function Box(low::T, high::T, shape::Tuple{Vararg{Int}}) where T<:Number
    return Box{T}(low*ones(T, shape), high*ones(T, shape), shape)
end

# Convenience constructors
Box(low::Array{T}, high::Array{T}) where T<:Number = Box{T}(low, high)

Base.ndims(space::UniformBox) = length(space.shape)
Base.eltype(::UniformBox{T}) where T = T

Base.ndims(space::Box) = length(space.shape)
Base.eltype(::Box{T}) where T = T

#TODO fix comparison of spaces
function Base.isequal(box1::Box{T1}, box2::Box{T2}) where {T1,T2}
    T1 == T2 && box1.low == box2.low && box1.high == box2.high && box1.shape == box2.shape
end

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


# Helper function to process actions: ensure correct type and clipping
function process_action(action::AbstractArray, action_space::UniformBox{T}) where T
    # First check if type conversion is needed
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    # Then clip to bounds
    action = clamp.(action, action_space.low, action_space.high)
    return action
end

# Extend Random.rand for Box spaces
"""
    rand([rng], space::Box{T})

Sample a random value from the box space with potentially different bounds per dimension.

# Examples
```julia
low = Float32[-1.0, -2.0]
high = Float32[1.0, 3.0]
space = Box(low, high)
sample = rand(space)  # Returns a 2-element Float32 array with values in [-1,1] and [-2,3] respectively
```
"""
function Random.rand(rng::AbstractRNG, space::Box{T}) where T
    # Generate random values in [0, 1] with correct type and shape
    unit_random = rand(rng, T, space.shape...)
    # Scale to [low, high] range element-wise
    return unit_random .* (space.high .- space.low) .+ space.low
end

# Default RNG version
Random.rand(space::Box) = rand(Random.default_rng(), space)

# Multiple samples version
"""
    rand([rng], space::Box{T}, n::Integer)

Sample `n` random values from the box space.

Returns an array where the last dimension has size `n`.
"""
function Random.rand(rng::AbstractRNG, space::Box{T}, n::Integer) where T
    # Generate random values with an extra dimension for n samples
    unit_random = rand(rng, T, space.shape..., n)
    # Scale to [low, high] range element-wise
    # Need to add dimensions to low/high to broadcast correctly
    low_expanded = reshape(space.low, space.shape..., 1)
    high_expanded = reshape(space.high, space.shape..., 1)
    return unit_random .* (high_expanded .- low_expanded) .+ low_expanded
end

Random.rand(space::Box, n::Integer) = rand(Random.default_rng(), space, n)

"""
    sample in space::Box{T}

Check if a sample is within the bounds of the box space.

# Examples
```julia
low = Float32[-1.0, -2.0]
high = Float32[1.0, 3.0]
space = Box(low, high)
Float32[0.5, 1.5] in space  # Returns true
Float32[1.5, 0.0] in space  # Returns false (first element out of bounds)

# Can also use ∈ symbol
@test action ∈ action_space
```
"""
function Base.in(sample, space::Box{T}) where T
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

    # Check bounds element-wise
    return all(space.low .<= sample .<= space.high)
end

# Helper function to process actions: ensure correct type and clipping for Box
function process_action(action::AbstractArray, action_space::Box{T}) where T
    # First check if type conversion is needed
    if eltype(action) != T
        @warn "Action type mismatch: $(eltype(action)) != $T"
        action = convert.(T, action)
    end
    # Then clip to bounds element-wise
    action = clamp.(action, action_space.low, action_space.high)
    return action
end
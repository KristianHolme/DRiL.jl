# Optimization utility functions (used by SAC)

function polyak_update!(target::AbstractArray{T}, source::AbstractArray{T}, tau::T) where {T <: AbstractFloat}
    target .= tau .* source .+ (one(T) - tau) .* target
    return nothing
end

function polyak_update!(target::ComponentArray{T}, source::ComponentArray{T}, tau::T) where {T <: AbstractFloat}
    for key in keys(target)
        target[key] .= tau .* source[key] .+ (one(T) - tau) .* target[key]
    end
    return nothing
end

function merge_params(a1::ComponentArray, a2::ComponentArray)
    a3 = copy(a1)
    for key in keys(a2)
        a3[key] = a2[key]
    end
    return a3
end


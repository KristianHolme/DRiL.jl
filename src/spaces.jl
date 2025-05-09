abstract type AbstractSpace end
abstract type AbstractBox <: AbstractSpace end

struct UniformBox <: AbstractBox
    type::Type
    low::Number
    high::Number
    shape::Tuple
end

Base.ndims(space::UniformBox) = length(space.shape)

    
abstract type AbstractBox <: AbstractSpace end

struct UniformBox <: AbstractBox
    type::Type
    low::Number
    high::Number
    shape::Tuple
end

    
struct NoTrainingLogger <: AbstractTrainingLogger end

set_step!(::NoTrainingLogger, ::Integer) = nothing
increment_step!(::NoTrainingLogger, ::Integer) = nothing
log_scalar!(::NoTrainingLogger, ::AbstractString, ::Real) = nothing
log_dict!(::NoTrainingLogger, ::AbstractDict{<:AbstractString, <:Any}) = nothing
log_dict!(::NoTrainingLogger, ::AbstractDict{Symbol, <:Any}) = nothing
log_dict!(::NoTrainingLogger, ::NamedTuple) = nothing
log_hparams!(::NoTrainingLogger, ::AbstractDict{<:AbstractString, <:Any}, ::AbstractVector{<:AbstractString}) = nothing
flush!(::NoTrainingLogger) = nothing
close!(::NoTrainingLogger) = nothing

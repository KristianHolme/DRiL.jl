module DRiL_WandbExt

using DRiL
using Wandb
import DRiL: AbstractTrainingLogger, set_step!, increment_step!, log_scalar!, log_dict!, flush!, close!

mutable struct WandbBackend <: AbstractTrainingLogger
    wb::Wandb.WandbLogger
    current_step::Int
end

Base.convert(::Type{AbstractTrainingLogger}, wb::Wandb.WandbLogger) = WandbBackend(wb, 0)

DRiL.set_step!(lg::WandbBackend, s::Integer) = (Δ = s - lg.current_step; lg.current_step = s; Δ != 0 && Wandb.increment_step!(lg.wb, Δ); s)
DRiL.increment_step!(lg::WandbBackend, Δ::Integer) = (lg.current_step += Δ; Wandb.increment_step!(lg.wb, Δ); lg.current_step)

function DRiL.log_scalar!(lg::WandbBackend, k::AbstractString, v::Real)
    Wandb.log(lg.wb, Dict(k => v))
    return nothing
end

function DRiL.log_dict!(lg::WandbBackend, kv::AbstractDict{<:AbstractString, <:Any})
    d = Dict{String, Any}()
    for (k, v) in kv
        d[string(k)] = v
    end
    Wandb.log(lg.wb, d)
    return nothing
end

DRiL.flush!(::WandbBackend) = nothing
function DRiL.close!(lg::WandbBackend)
    try
        close(lg.wb)
    catch
    end
    return nothing
end

end

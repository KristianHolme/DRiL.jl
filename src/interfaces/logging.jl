# ------------------------------------------------------------
# Logging interface
# ------------------------------------------------------------
abstract type AbstractTrainingLogger end

"""
    set_step!(logger::AbstractTrainingLogger, step::Integer)

Set the global step for subsequent metric logs.
"""
function set_step! end
function increment_step! end

"""
    log_scalar!(logger::AbstractTrainingLogger, key::AbstractString, value::Real)

Log a single scalar metric under `key`.
"""
function log_scalar! end

function log_scalar!(logger::AbstractTrainingLogger, key::Symbol, value::Real)
    return log_scalar!(logger, string(key), value)
end

"""
    log_dict!(logger::AbstractTrainingLogger, kv)

Log multiple metrics at once from a key/value payload.

Backends are required to support `AbstractDict{<:AbstractString,<:Any}`.
The interface additionally normalizes and accepts:

- `AbstractDict{<:Symbol,<:Any}`
- `NamedTuple`
- `Base.Pairs` (e.g. the result of `pairs((; a = 1, b = 2))`)
- `Tuple{Vararg{Pair}}` and `AbstractVector{<:Pair}`
"""
function log_dict! end

function _string_key_dict_from_pairs(kv_pairs)
    out = Dict{String, Any}()
    for (k, v) in kv_pairs
        out[string(k)] = v
    end
    return out
end

function _stringify_keys(kv::AbstractDict{<:Symbol, <:Any})
    return _string_key_dict_from_pairs(kv)
end

function _stringify_keys(kv::NamedTuple)
    return _string_key_dict_from_pairs(pairs(kv))
end

function _stringify_keys(kv::Base.Pairs)
    return _string_key_dict_from_pairs(kv)
end

function _stringify_keys(kv::Tuple{Vararg{Pair}})
    return _string_key_dict_from_pairs(kv)
end

function _stringify_keys(kv::AbstractVector{<:Pair})
    return _string_key_dict_from_pairs(kv)
end

function log_dict!(logger::AbstractTrainingLogger, kv::AbstractDict{<:Symbol, <:Any})
    return log_dict!(logger, _stringify_keys(kv))
end

function log_dict!(logger::AbstractTrainingLogger, kv::NamedTuple)
    return log_dict!(logger, _stringify_keys(kv))
end

function log_dict!(logger::AbstractTrainingLogger, kv::Base.Pairs)
    return log_dict!(logger, _stringify_keys(kv))
end

function log_dict!(logger::AbstractTrainingLogger, kv::Tuple{Vararg{Pair}})
    return log_dict!(logger, _stringify_keys(kv))
end

function log_dict!(logger::AbstractTrainingLogger, kv::AbstractVector{<:Pair})
    return log_dict!(logger, _stringify_keys(kv))
end

"""
    log_hparams!(logger::AbstractTrainingLogger, hparams::AbstractDict{<:AbstractString,<:Any}, metrics::AbstractVector{<:AbstractString})

Write hyperparameters and associate them with specified metrics for hyperparameter tuning.
"""
function log_hparams! end

function log_hparams!(
        logger::AbstractTrainingLogger,
        hparams::AbstractDict{<:Symbol, <:Any},
        metrics::AbstractVector{<:AbstractString}
    )
    return log_hparams!(logger, _stringify_keys(hparams), metrics)
end

function log_hparams!(
        logger::AbstractTrainingLogger,
        hparams::NamedTuple,
        metrics::AbstractVector{<:AbstractString}
    )
    return log_hparams!(logger, _stringify_keys(hparams), metrics)
end

"""
    flush!(logger::AbstractTrainingLogger)

Ensure any buffered data is pushed to the backend. Implementations may no-op.
"""
function flush! end

"""
    close!(logger::AbstractTrainingLogger)

Finalize the logger and release resources. Implementations may no-op.
"""
function close! end

# Strict conversion gate: identity for wrapper, error otherwise
Base.convert(::Type{AbstractTrainingLogger}, x::AbstractTrainingLogger) = x
Base.convert(::Type{AbstractTrainingLogger}, x) = error(
    "Unsupported logger $(typeof(x)). Pass NoTrainingLogger(), TensorBoardLogger.TBLogger, Wandb.WandbLogger, or a DearDiary experiment ID."
)


"""
    log_stats(env::AbstractEnv, logger::AbstractLogger) -> Nothing

Log environment-specific statistics to a logger (optional interface).

# Arguments
- `env::AbstractEnv`: The environment
- `logger::AbstractLogger`: The logger to write to

# Returns
- `Nothing`

# Notes
Default implementation does nothing. Environments can override to log custom metrics.
"""
function log_stats(env::AbstractEnv, logger::AbstractTrainingLogger)
    return nothing
end

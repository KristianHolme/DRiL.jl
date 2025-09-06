struct TrainingLogger
    tb_logger::Union{Nothing, TBLogger}
    progress_meter::Union{Nothing, Progress}
    latest_metrics::Dict{String, Any}
end

function TrainingLogger(; logdir::Union{Nothing, String} = nothing, verbose::Int = 0)
    tb_logger = isnothing(logdir) ? nothing : TensorBoardLogger(logdir)
    progress_meter = nothing
    latest_metrics = Dict{String, Any}()
    return TrainingLogger(tb_logger, progress_meter, latest_metrics)
end

function init_training!(logger::TrainingLogger, total_steps::Int)
    progress_meter = Progress(
        total_steps, desc = "Training...",
        showspeed = true, enabled = logger.verbose > 0
    )
    logger.progress_meter = progress_meter
    return nothing
end

function get_showvalues(logger::TrainingLogger)
    showvalues = []
    for (key, value) in logger.latest_metrics
        push!(showvalues, (key, value))
    end
    return showvalues
end

showvalues_func(logger::TrainingLogger) = () -> get_showvalues(logger)

function update_progress!(logger::TrainingLogger, step::Int = 1)
    if !isnothing(logger.progress_meter)
        next!(logger.progress_meter; step, showvalues = get_showvalues(logger))
    end
    return nothing
end

function finish_training!(logger::TrainingLogger)
    if !isnothing(logger.progress_meter)
        finish!(logger.progress_meter)
    end
    return nothing
end

function log_value(logger::TrainingLogger, key::String, value::Any)
    return logger.latest_metrics[key] = value
end

function log_metrics!(logger::TrainingLogger, metrics::Dict{String, Any})
    for (key, value) in metrics
        log_value(logger, key, value)
    end
    return nothing
end

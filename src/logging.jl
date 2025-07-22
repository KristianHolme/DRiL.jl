function TensorBoardLogger.write_hparams!(logger::TBLogger, alg::AbstractAlgorithm, metrics::AbstractArray{String})
    hparams = get_hparams(alg)
    TensorBoardLogger.write_hparams!(logger, hparams, metrics)
    nothing
end

function TensorBoardLogger.write_hparams!(logger::TBLogger, agent::AbstractAgent, metrics::AbstractArray{String})
    hparams = get_hparams(agent)
    TensorBoardLogger.write_hparams!(logger, hparams, metrics)
    nothing
end

#FIXME: is this piracy?
function TensorBoardLogger.write_hparams!(logger::TBLogger, alg::AbstractAlgorithm, agent::AbstractAgent, metrics::AbstractArray{String})
    hparams = merge(get_hparams(alg), get_hparams(agent))
    TensorBoardLogger.write_hparams!(logger, hparams, metrics)
    nothing
end

function get_hparams(alg::AbstractAlgorithm)
    @warn "get_hparams is not implemented for $(typeof(alg)). No hyperparameters will be logged."
    return Dict{String,Any}()
end

function get_hparams(agent::AbstractAgent)
    @warn "get_hparams is not implemented for $(typeof(agent)). No hyperparameters will be logged."
    return Dict{String,Any}()
end

function get_hparams(alg::PPO)
    hparams = Dict{String,Any}(
        "gamma" => alg.gamma,
        "gae_lambda" => alg.gae_lambda,
        "clip_range" => alg.clip_range,
        "ent_coef" => alg.ent_coef,
        "vf_coef" => alg.vf_coef,
        "max_grad_norm" => alg.max_grad_norm,
        "normalize_advantage" => alg.normalize_advantage
    )

    if !isnothing(alg.clip_range_vf)
        hparams["clip_range_vf"] = alg.clip_range_vf
    end

    if !isnothing(alg.target_kl)
        hparams["target_kl"] = alg.target_kl
    end

    return hparams
end

function get_hparams(agent::ActorCriticAgent)
    hparams = Dict{String,Any}(
        "learning_rate" => agent.learning_rate,
        "batch_size" => agent.batch_size,
        "n_steps" => agent.n_steps,
        "epochs" => agent.epochs)
    return hparams
end
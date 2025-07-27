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
        "normalize_advantage" => alg.normalize_advantage,
        "learning_rate" => alg.learning_rate,
        "batch_size" => alg.batch_size,
        "n_steps" => alg.n_steps,
        "epochs" => alg.epochs
    )

    if !isnothing(alg.clip_range_vf)
        hparams["clip_range_vf"] = alg.clip_range_vf
    end

    if !isnothing(alg.target_kl)
        hparams["target_kl"] = alg.target_kl
    end

    return hparams
end

function get_hparams(alg::SAC)
    hparams = Dict{String,Any}(
        "learning_rate" => alg.learning_rate,
        "buffer_size" => alg.buffer_size,
        "start_steps" => alg.start_steps,
        "batch_size" => alg.batch_size,
        "tau" => alg.tau,
        "gamma" => alg.gamma,
        "train_freq" => alg.train_freq,
        "gradient_steps" => alg.gradient_steps,
        "ent_coef" => string(alg.ent_coef),
        "target_update_interval" => alg.target_update_interval
    )
    return hparams
end

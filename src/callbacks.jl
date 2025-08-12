function on_training_start(callback::AbstractCallback, locals::Dict)
    return true
end
function on_rollout_start(callback::AbstractCallback, locals::Dict)
    return true
end
function on_step(callback::AbstractCallback, locals::Dict)
    return true
end
function on_rollout_end(callback::AbstractCallback, locals::Dict)
    return true
end
function on_training_end(callback::AbstractCallback, locals::Dict)
    return true
end

@testitem "NoTrainingLogger basics" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_scalar!(lg, :a_sym, 1.0)
    log_dict!(lg, Dict("b" => 2.0))
    log_dict!(lg, Dict(:b_sym => 2.0))
    log_dict!(lg, (b_nt = 2.0,))
    log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["metric1"])
    log_hparams!(lg, Dict(:lr => 0.01, :gamma => 0.99), ["metric1"])
    log_hparams!(lg, (lr = 0.01, gamma = 0.99), ["metric1"])
    flush!(lg)
    close!(lg)
    @test true
end

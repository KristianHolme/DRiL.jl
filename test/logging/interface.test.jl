@testitem "NoTrainingLogger basics" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_dict!(lg, Dict("b" => 2.0))
    flush!(lg)
    close!(lg)
    @test true
end

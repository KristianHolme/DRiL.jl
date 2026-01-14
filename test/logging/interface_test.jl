@testitem "NoTrainingLogger basics" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    log_scalar!(lg, "a", 1.0)
    log_dict!(lg, Dict("b" => 2.0))
    log_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["metric1"])
    flush!(lg)
    close!(lg)
    @test true
end

@testitem "NoTrainingLogger symbol-keyed dict" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    # Symbol-keyed dict should work
    log_dict!(lg, Dict(:loss => 0.5, :accuracy => 0.95))
    @test true
end

@testitem "NoTrainingLogger named tuple" begin
    lg = NoTrainingLogger()
    set_step!(lg, 10)
    # Named tuple should work
    log_dict!(lg, (loss = 0.5, accuracy = 0.95))
    @test true
end

@testitem "log_dict! with symbol keys conversion" begin
    using DRiL: _to_string_keyed_dict

    # Test symbol-keyed dict conversion
    symbol_dict = Dict(:loss => 0.5, :accuracy => 0.95)
    string_dict = _to_string_keyed_dict(symbol_dict)
    @test string_dict isa Dict{String, Float64}
    @test string_dict["loss"] == 0.5
    @test string_dict["accuracy"] == 0.95
end

@testitem "log_dict! with named tuple conversion" begin
    using DRiL: _to_string_keyed_dict

    # Test named tuple conversion
    nt = (loss = 0.5, accuracy = 0.95, epochs = 10)
    string_dict = _to_string_keyed_dict(nt)
    @test string_dict isa Dict{String, Any}
    @test string_dict["loss"] == 0.5
    @test string_dict["accuracy"] == 0.95
    @test string_dict["epochs"] == 10
end

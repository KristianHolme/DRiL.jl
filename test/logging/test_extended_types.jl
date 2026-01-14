using DRiL
using Test
using TestItems

@testitem "Extended Logging Types" begin
    # Mock logger to verify calls
    struct MockLogger <: AbstractTrainingLogger
        logs::Dict{String, Any}
    end
    MockLogger() = MockLogger(Dict{String, Any}())
    
    # We need to implement the interface for MockLogger
    DRiL.set_step!(::MockLogger, ::Integer) = nothing
    DRiL.increment_step!(::MockLogger, ::Integer) = nothing
    DRiL.log_scalar!(l::MockLogger, k::AbstractString, v::Real) = (l.logs[k] = v; nothing)
    
    function DRiL.log_dict!(l::MockLogger, kv::AbstractDict{<:AbstractString, <:Any})
        for (k, v) in kv
            l.logs[k] = v
        end
    end
    
    DRiL.log_hparams!(::MockLogger, ::AbstractDict, ::AbstractVector) = nothing
    DRiL.flush!(::MockLogger) = nothing
    DRiL.close!(::MockLogger) = nothing

    logger = MockLogger()

    # Test NamedTuple
    log_dict!(logger, (a=1, b=2.0))
    @test logger.logs["a"] == 1
    @test logger.logs["b"] == 2.0

    # Test Dict{Symbol, Any}
    log_dict!(logger, Dict(:c => 3, :d => 4.0))
    @test logger.logs["c"] == 3
    @test logger.logs["d"] == 4.0
    
    # Test mixed Dict{Symbol, Any} with other types
    log_dict!(logger, Dict{Symbol, Any}(:e => 5, :f => "ignore"))
    @test logger.logs["e"] == 5
    @test logger.logs["f"] == "ignore"
end

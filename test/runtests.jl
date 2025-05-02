using DRiL
using Test
using Aqua
using JET

@testset "DRiL.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DRiL)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DRiL; target_defined_modules = true)
    end
    # Write your tests here.
end

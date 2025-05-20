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

# using Test
# include("test_buffers.jl")
# include("test_policies.jl")
# include("test_agents.jl")
# include("test_algorithms.jl")
# include("test_environments.jl")
# include("test_utils.jl")
# include("test_spaces.jl")
# include("test_basic_types.jl")

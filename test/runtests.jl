using DRiL
using Test
using Aqua
using JET

@testitem "Code quality (Aqua.jl)" begin
    Aqua.test_all(DRiL)
end

@testitem "Code linting (JET.jl)" begin
    JET.test_package(DRiL; target_defined_modules=true)
end

@testitem "buffers.jl" begin
    include("test_buffers.jl")
end

# @testitem "policies.jl" begin
#     include("test_policies.jl")
# end
# @testitem "agents.jl" begin
#     include("test_agents.jl")
# end
# @testitem "algorithms.jl" begin
#     include("test_algorithms.jl")
# end
# @testitem "environments.jl" begin
#     include("test_environments.jl")
# end
# @testitem "utils.jl" begin
#     include("test_utils.jl")
# end
# @testitem "spaces.jl" begin
#     include("test_spaces.jl")
# end
# @testitem "basic_types.jl" begin
#     include("test_basic_types.jl")
# end

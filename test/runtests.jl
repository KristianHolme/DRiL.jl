using DRiL
using Test
using TestItems
using TestItemRunner

# Include the shared test setup module
# include("test_shared_setup.jl")

# Quality assurance tests
@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua
    Aqua.test_all(DRiL)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET
    JET.test_package(DRiL; target_defined_modules=true)
end

# Include all test files (they contain @testitem definitions)
# include("test_env_validation.jl")
# include("test_buffers.jl")
# include("test_gae.jl")
# include("test_spaces.jl")
# include("test_normalize_wrapper.jl")

# Run all tests
@run_package_tests

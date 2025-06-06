using DRiL
using Test
using TestItems
using TestItemRunner

# Quality assurance tests
@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua
    Aqua.test_all(DRiL)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET
    JET.test_package(DRiL; target_defined_modules=true)
end

# Run all tests
@run_package_tests

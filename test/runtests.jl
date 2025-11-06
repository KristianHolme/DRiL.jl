using TestItemRunner

# Quality assurance tests
@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua
    Aqua.test_all(DRiL)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    using JET
    if get(ENV, "JET_STRICT", "") == "1"
        JET.test_package(DRiL; target_modules = (DRiL,))
    else
        # Advisory mode: print report but don't fail CI
        report = JET.report_package(DRiL; target_modules = (DRiL,), toplevel_logger = nothing)
        println(report)
        @test true
    end
end

# Run all tests
@run_package_tests

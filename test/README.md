# DRiL.jl Test Suite

This test suite uses the [TestItem framework](https://www.julia-vscode.org/docs/stable/userguide/testing/) for modern, modular testing in Julia. The framework allows individual test execution and better test organization.

## Test Structure

### Core Components

- **`test_shared_setup.jl`**: Contains the `@testmodule SharedTestSetup` with shared environments, policies, and utilities used across multiple test items
- **`runtests.jl`**: Main entry point that includes all test files and runs the test suite

### Test Files

1. **`test_buffers.jl`**: Buffer functionality tests
   - Buffer logprobs consistency
   - Buffer reset functionality  
   - Buffer trajectory bootstrap handling
   - Buffer data integrity

2. **`test_gae.jl`**: Generalized Advantage Estimation tests
   - GAE computation analytical verification
   - GAE computation with different parameters (γ, λ)
   - GAE computation with custom environments
   - Multiple episodes handling
   - Infinite horizon environments
   - Edge cases (single step, zero parameters)

3. **`test_env_validation.jl`**: Environment interface validation
   - Environment interface compliance
   - Episode completion behavior
   - Wrapper functionality
   - Space constraint adherence
   - Reproducibility testing

4. **`test_spaces.jl`**: Space interface tests
   - UniformBox creation and properties
   - Random sampling (`Random.rand` extensions)
   - Containment checking (`Base.in` extensions)
   - Type consistency
   - Interface completeness

## Test Tags

Tests are organized with tags for easy filtering:

- **`:buffers`**: Buffer-related functionality (excluding GAE computation)
- **`:gae`**: GAE computation tests
- **`:algorithms`**: Algorithm implementation tests
- **`:environments`**: Environment interface and validation
- **`:spaces`**: Space interface and operations
- **`:validation`**: Interface compliance tests
- **`:random`**: Random sampling and reproducibility
- **`:edge_cases`**: Edge case and boundary testing
- **`:quality`**: Code quality (Aqua.jl, JET.jl)
- **`:analytical`**: Tests with analytical verification
- **`:parametric`**: Tests with multiple parameter combinations
- **`:integrity`**: Data integrity and consistency tests

## Running Tests

### All Tests
```julia
julia> using Pkg; Pkg.test()
```

### Individual Test Items (VS Code)
In VS Code with the Julia extension:
1. Individual `@testitem` blocks can be run using the ▶️ button next to each test
2. Use the Testing activity bar to see all tests and their status
3. Filter tests by tags in the VS Code UI

### Command Line with Filtering
```julia
# Run only buffer tests (excludes GAE computation)
@run_package_tests filter=ti->(:buffers in ti.tags)

# Run only GAE tests  
@run_package_tests filter=ti->(:gae in ti.tags)

# Run only environment tests
@run_package_tests filter=ti->(:environments in ti.tags)

# Run only space tests
@run_package_tests filter=ti->(:spaces in ti.tags)

# Skip quality tests
@run_package_tests filter=ti->!(:quality in ti.tags)

# Run specific test by name
@run_package_tests filter=ti->(ti.name == "GAE computation analytical verification")
```

## Shared Test Setup

The `SharedTestSetup` module provides:

### Test Environments
- **`CustomEnv`**: Gives reward 1.0 only at final timestep (for GAE testing)
- **`InfiniteHorizonEnv`**: Always gives reward 1.0, never terminates
- **`SimpleRewardEnv`**: Basic environment for general testing
- **`ConstantObsWrapper`**: Environment wrapper for consistent observations

### Test Policies
- **`ConstantValuePolicy`**: Returns constant values for predictable testing

### Utilities
- **`compute_expected_gae`**: Analytical GAE computation for verification
- **`AbstractEnvWrapper`**: Base type for environment wrappers

## Key Features

1. **Modular Design**: Each test item is independent and can be run individually
2. **Shared Setup**: Common environments and policies defined once in `@testmodule`
3. **Clear Separation**: Buffer tests focus on buffer functionality, GAE tests focus on GAE computation
4. **Tagging System**: Tests are categorized for easy filtering and organization
5. **Analytical Verification**: GAE tests use analytical computation for correctness
6. **Edge Case Coverage**: Comprehensive testing including boundary conditions
7. **Interface Validation**: Ensures proper implementation of DRiL interfaces
8. **Linting Compliance**: Code follows Julia best practices (uses `eachindex` instead of `length` for indexing)

## Test Organization Philosophy

- **`test_buffers.jl`**: Tests buffer data structures, reset functionality, trajectory handling, and data integrity
- **`test_gae.jl`**: Tests GAE algorithm correctness with analytical verification across different parameter combinations
- **`test_env_validation.jl`**: Tests environment interface compliance and behavior consistency
- **`test_spaces.jl`**: Tests space interface extensions and mathematical properties

## Adding New Tests

1. Create `@testitem` blocks with descriptive names and appropriate tags
2. Use `setup=[SharedTestSetup]` to access shared test environments and utilities
3. Add tags that describe the test category and functionality
4. Include the test file in `runtests.jl`
5. Document any new shared utilities in `SharedTestSetup`
6. Follow Julia best practices (use `eachindex` for iteration, avoid `length()` for indexing)

## Best Practices

- Use meaningful test names that describe what is being tested
- Add appropriate tags for test organization
- Leverage shared setup to avoid code duplication
- Include both positive and negative test cases
- Test edge cases and boundary conditions
- Use analytical verification where possible for numerical computations
- Follow Julia indexing best practices (`eachindex` over `1:length(...)`)
- Separate concerns: buffer tests should focus on buffers, GAE tests on GAE computation 
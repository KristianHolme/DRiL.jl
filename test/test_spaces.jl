@testitem "UniformBox space creation and properties" tags = [:spaces, :basic] begin
    using Random

    # Test basic UniformBox creation
    low = -1.0f0
    high = 1.0f0
    shape = (2,)
    space = UniformBox{Float32}(low, high, shape)

    @test typeof(space) == UniformBox{Float32}
    @test space.low == low
    @test space.high == high
    @test space.shape == shape

    # Test single dimension space
    space_1d = UniformBox{Float32}(-1.0f0, 1.0f0, (1,))
    @test space_1d.shape == (1,)
    @test space_1d.low == -1.0f0
    @test space_1d.high == 1.0f0

    # Test higher dimensional space
    space_3d = UniformBox{Float32}(-1.0f0, 1.0f0, (3,))
    @test space_3d.shape == (3,)
    @test space_3d.low == -1.0f0
    @test space_3d.high == 1.0f0
end

@testitem "Random sampling from UniformBox" tags = [:spaces, :random] begin
    using Random

    # Test Random.rand extension
    space = UniformBox{Float32}(-2.0f0, 2.0f0, (2,))
    rng = MersenneTwister(42)

    # Test rand(rng, space)
    sample1 = rand(rng, space)
    @test length(sample1) == 2
    @test space.low ≤ sample1[1] ≤ space.high
    @test space.low ≤ sample1[2] ≤ space.high
    @test eltype(sample1) == Float32

    # Test rand(space) - default RNG
    sample2 = rand(space)
    @test length(sample2) == 2
    @test space.low ≤ sample2[1] ≤ space.high
    @test space.low ≤ sample2[2] ≤ space.high
    @test eltype(sample2) == Float32

    # Test rand(rng, space, n) - multiple samples
    n = 5
    samples = rand(rng, space, n)
    @test size(samples) == (2, n)
    for i in 1:n
        @test space.low ≤ samples[1, i] ≤ space.high
        @test space.low ≤ samples[2, i] ≤ space.high
    end
    @test eltype(samples) == Float32

    # Test rand(space, n) - multiple samples with default RNG
    samples2 = rand(space, n)
    @test size(samples2) == (2, n)
    @test eltype(samples2) == Float32
end

@testitem "Space containment checking" tags = [:spaces, :containment] begin
    using Random

    # Test Base.in extension
    space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))

    # Test valid samples are in space
    valid_sample = [0.5f0, 0.8f0]
    @test valid_sample ∈ space
    @test valid_sample in space

    # Test boundary values
    @test [-1.0f0, -1.0f0] ∈ space  # Lower boundary
    @test [1.0f0, 1.0f0] ∈ space    # Upper boundary

    # Test invalid samples are not in space
    invalid_samples = [
        [1.5f0, 0.0f0],    # First component too high
        [0.0f0, -1.5f0],   # Second component too low
        [-1.5f0, 1.5f0],   # Both components out of bounds
        [0.0f0, 0.0f0, 0.0f0]  # Wrong dimensions
    ]

    for sample in invalid_samples
        @test !(sample ∈ space)
        @test !(sample in space)
    end

    # Test with generated samples
    rng = MersenneTwister(123)
    for i in 1:10
        sample = rand(rng, space)
        @test sample ∈ space
    end
end

@testitem "Space containment edge cases" tags = [:spaces, :containment, :edge_cases] begin
    # Test edge cases for space containment

    # Single dimension space
    space_1d = UniformBox{Float32}(-1.0f0, 1.0f0, (1,))
    @test [0.0f0] ∈ space_1d
    @test [1.1f0] ∉ space_1d
    @test [-1.1f0] ∉ space_1d

    # Test with different numeric types
    @test [0] ∉ space_1d  # Int should not work
    @test [0.0] ∉ space_1d  # Float64 should not work

    # Test empty arrays
    @test !([Float32[]] ∈ space_1d)

    # Test very small space
    tiny_space = UniformBox{Float32}(0.0f0, 1f-6, (1,))
    @test [0.0f0] ∈ tiny_space
    @test [1f-6] ∈ tiny_space
    @test [1f-5] ∉ tiny_space
end

@testitem "Space random sampling properties" tags = [:spaces, :random, :properties] begin
    using Random
    using Statistics

    # Test that samples are uniformly distributed
    space = UniformBox{Float32}(-2.0f0, 2.0f0, (2,))
    rng = MersenneTwister(42)

    n_samples = 1000
    samples = rand(rng, space, n_samples)

    # Check all samples are in bounds
    for i in 1:n_samples
        sample = samples[:, i]
        @test sample ∈ space
    end

    # Check approximate uniform distribution (loose bounds due to randomness)
    mean_vals = mean(samples, dims=2)
    expected_mean = (space.high + space.low) / 2

    # With 1000 samples, means should be reasonably close to expected
    @test abs(mean_vals[1] - expected_mean) < 0.2
    @test abs(mean_vals[2] - expected_mean) < 0.2

    # Check that we get different samples
    @test length(unique(samples[1, :])) > n_samples * 0.9  # Most samples should be unique
end

@testitem "Space type consistency" tags = [:spaces, :types] begin
    using Random
    # Test type consistency across different operations

    # Float32 space
    space_f32 = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))
    sample_f32 = rand(space_f32)
    @test eltype(sample_f32) == Float32
    @test sample_f32 ∈ space_f32

    # Test multiple samples maintain type
    samples_f32 = rand(space_f32, 5)
    @test eltype(samples_f32) == Float32
    @test size(samples_f32) == (2, 5)

    # Test that wrong type doesnt work
    @test [0, 0] ∉ space_f32

    # Test boundary behavior with exact type matching
    @test [-1.0f0, 1.0f0] ∈ space_f32
    @test [1.0f0, -1.0f0] ∈ space_f32
end

@testitem "Space interface completeness" tags = [:spaces, :interface] begin
    using Random

    # Test that UniformBox implements all expected interface methods
    space = UniformBox{Float32}(-1.0f0, 1.0f0, (2,))

    # Test that all expected methods exist
    @test hasmethod(rand, (AbstractRNG, typeof(space)))
    @test hasmethod(rand, (typeof(space),))
    @test hasmethod(rand, (AbstractRNG, typeof(space), Int))
    @test hasmethod(rand, (typeof(space), Int))
    @test hasmethod(in, (Vector{Float32}, typeof(space)))

    # Test that methods work as expected
    rng = MersenneTwister(42)

    # Single sample methods
    s1 = rand(rng, space)
    s2 = rand(space)
    @test s1 ∈ space
    @test s2 ∈ space

    # Multiple sample methods
    s3 = rand(rng, space, 3)
    s4 = rand(space, 3)
    @test size(s3) == (2, 3)
    @test size(s4) == (2, 3)
    @test all(s3[:, i] ∈ space for i in 1:3)
    @test all(s4[:, i] ∈ space for i in 1:3)
end
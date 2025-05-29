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

# Tests for the general Box space with different bounds per dimension
@testitem "Box space creation and properties" tags = [:spaces, :basic, :box] begin
    using Random

    # Test basic Box creation with different bounds per dimension
    low = Float32[-2.0, -1.0]
    high = Float32[1.0, 3.0]
    space = Box(low, high)

    @test typeof(space) == Box{Float32}
    @test space.low == low
    @test space.high == high
    @test space.shape == (2,)

    # Test single dimension Box
    space_1d = Box(Float32[-5.0], Float32[10.0])
    @test space_1d.shape == (1,)
    @test space_1d.low == Float32[-5.0]
    @test space_1d.high == Float32[10.0]

    # Test higher dimensional Box with varied bounds
    low_3d = Float32[-1.0, 0.0, -10.0]
    high_3d = Float32[1.0, 5.0, 0.0]
    space_3d = Box(low_3d, high_3d)
    @test space_3d.shape == (3,)
    @test space_3d.low == low_3d
    @test space_3d.high == high_3d
end

@testitem "Box space validation" tags = [:spaces, :validation, :box] begin
    # Test Box constructor validation

    # Should work - valid bounds
    @test Box(Float32[-1.0, -2.0], Float32[1.0, 2.0]) isa Box{Float32}

    # Should fail - mismatched shapes
    @test_throws AssertionError Box(Float32[-1.0], Float32[1.0, 2.0])

    # Should fail - low > high in some dimension
    @test_throws AssertionError Box(Float32[1.0, -1.0], Float32[0.0, 1.0])

    # Edge case - low == high (valid)
    space_equal = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test space_equal isa Box{Float32}
end

@testitem "Random sampling from Box" tags = [:spaces, :random, :box] begin
    using Random

    # Test Box with different bounds per dimension
    low = Float32[-3.0, 0.0]
    high = Float32[2.0, 10.0]
    space = Box(low, high)
    rng = MersenneTwister(42)

    # Test rand(rng, space)
    sample1 = rand(rng, space)
    @test length(sample1) == 2
    @test low[1] ≤ sample1[1] ≤ high[1]
    @test low[2] ≤ sample1[2] ≤ high[2]
    @test eltype(sample1) == Float32

    # Test rand(space) - default RNG
    sample2 = rand(space)
    @test length(sample2) == 2
    @test low[1] ≤ sample2[1] ≤ high[1]
    @test low[2] ≤ sample2[2] ≤ high[2]
    @test eltype(sample2) == Float32

    # Test rand(rng, space, n) - multiple samples
    n = 5
    samples = rand(rng, space, n)
    @test size(samples) == (2, n)
    for i in 1:n
        @test low[1] ≤ samples[1, i] ≤ high[1]
        @test low[2] ≤ samples[2, i] ≤ high[2]
    end
    @test eltype(samples) == Float32

    # Test rand(space, n) - multiple samples with default RNG
    samples2 = rand(space, n)
    @test size(samples2) == (2, n)
    @test eltype(samples2) == Float32
end

@testitem "Box containment checking" tags = [:spaces, :containment, :box] begin
    using Random

    # Test Box with asymmetric bounds
    low = Float32[-2.0, 1.0]
    high = Float32[0.0, 5.0]
    space = Box(low, high)

    # Test valid samples are in space
    valid_samples = [
        Float32[-1.0, 3.0],
        Float32[-2.0, 1.0],  # Lower boundary
        Float32[0.0, 5.0],   # Upper boundary
        Float32[-1.5, 2.5]
    ]

    for sample in valid_samples
        @test sample ∈ space
        @test sample in space
    end

    # Test invalid samples are not in space
    invalid_samples = [
        Float32[0.5, 3.0],    # First component too high
        Float32[-1.0, 0.5],   # Second component too low
        Float32[-3.0, 6.0],   # Both components out of bounds
        Float32[-1.0, 3.0, 0.0]  # Wrong dimensions
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

@testitem "Box edge cases and special configurations" tags = [:spaces, :box, :edge_cases] begin
    # Test very small bounds differences
    tiny_space = Box(Float32[0.0], Float32[1e-6])
    @test Float32[0.0] ∈ tiny_space
    @test Float32[1e-6] ∈ tiny_space
    @test !(Float32[1e-5] ∈ tiny_space)

    # Test negative bounds
    neg_space = Box(Float32[-10.0, -5.0], Float32[-1.0, -2.0])
    @test Float32[-5.0, -3.0] ∈ neg_space
    @test !(Float32[0.0, -3.0] ∈ neg_space)

    # Test mixed positive/negative bounds
    mixed_space = Box(Float32[-1.0, 2.0], Float32[1.0, 8.0])
    @test Float32[0.0, 5.0] ∈ mixed_space
    @test !(Float32[2.0, 1.0] ∈ mixed_space)

    # Test single point space (low == high)
    point_space = Box(Float32[1.0, 2.0], Float32[1.0, 2.0])
    @test Float32[1.0, 2.0] ∈ point_space
    @test !(Float32[1.0, 2.1] ∈ point_space)
end

@testitem "Box vs UniformBox equivalence" tags = [:spaces, :box, :equivalence] begin
    using Random

    # Test that Box with uniform bounds behaves like UniformBox
    low_val = -1.5f0
    high_val = 2.5f0
    shape = (3,)

    uniform_box = UniformBox{Float32}(low_val, high_val, shape)
    general_box = Box(fill(low_val, shape[1]), fill(high_val, shape[1]))

    # Test properties match
    @test uniform_box.shape == general_box.shape
    @test all(uniform_box.low .== general_box.low)
    @test all(uniform_box.high .== general_box.high)

    # Test sampling produces equivalent results
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)

    sample_uniform = rand(rng1, uniform_box)
    sample_general = rand(rng2, general_box)

    # Should be approximately equal (same RNG seed)
    @test sample_uniform ≈ sample_general

    # Test containment equivalence
    test_samples = [
        Float32[-1.0, 0.0, 1.0],
        Float32[3.0, 0.0, 0.0],  # Out of bounds
        Float32[-2.0, 0.0, 0.0]  # Out of bounds
    ]

    for sample in test_samples
        @test (sample ∈ uniform_box) == (sample ∈ general_box)
    end
end

@testitem "Box interface completeness" tags = [:spaces, :interface, :box] begin
    using Random

    # Test that Box implements all expected interface methods
    space = Box(Float32[-1.0, 0.0], Float32[1.0, 5.0])

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
@testitem "DiagGaussian vs Distributions.MvNormal" begin
    using Random
    using Distributions
    using LinearAlgebra
    using DRiL.DRiLDistributions

    same_outputs = Bool[]

    for i in 1:100
        mean = rand(Float32, 2, 2)
        log_std = rand(Float32, 2, 2)

        flat_mean = vec(mean)
        flat_log_std = vec(log_std)
        mvn = MvNormal(flat_mean, LinearAlgebra.Diagonal(map(abs2, exp.(flat_log_std))))

        d = DiagGaussian(mean, log_std)
        x = rand(Float32, 2, 2)

        flat_x = vec(x)

        custom_logpdf = DRiLDistributions.logpdf(d, x)
        dist_logpdf = Distributions.logpdf(mvn, flat_x)
        push!(same_outputs, custom_logpdf ≈ dist_logpdf)

        custom_entropy = DRiLDistributions.entropy(d)
        dist_entropy = Distributions.entropy(mvn)
        push!(same_outputs, custom_entropy ≈ dist_entropy)
    end

    @test all(same_outputs)
end

@testitem "Categorical vs Distributions.Categorical" begin
    using Random
    using Distributions

    same_outputs = Bool[]

    for N in [3,8], i in 1:100
        p = rand(Float32, N)
        p = p ./ sum(p)

        d = DRiLDistributions.Categorical(p)

        dist_d = Distributions.Categorical(p)

        custom_logpdf = DRiLDistributions.logpdf(d, 1)
        dist_logpdf = Distributions.logpdf(dist_d, 1)
        push!(same_outputs, custom_logpdf ≈ dist_logpdf)

        custom_entropy = DRiLDistributions.entropy(d)
        dist_entropy = Distributions.entropy(dist_d)
        push!(same_outputs, custom_entropy ≈ dist_entropy)
    end

    @test all(same_outputs)
end


@testitem "Diaggaussion constructor" begin
    using Random

    same_outputs = Bool[]

    #test strict types
    @test_throws MethodError DiagGaussian([1.0], [2f0])

    d = DiagGaussian([1.0f0], [2f0])
    @test_throws MethodError DRiLDistributions.logpdf(d, [1.0])


    mean_batch = rand(Float32, 2, 2, 7)
    std_batch = rand(Float32, 2, 2, 7)

    x_batch = rand(Float32, 2, 2, 7)

    @test begin
        ds = DiagGaussian.(eachslice(mean_batch, dims=ndims(mean_batch)), eachslice(std_batch, dims=ndims(std_batch)))
        entropies = DRiLDistributions.entropy.(ds)
        logpdfs = DRiLDistributions.logpdf.(ds, eachslice(x_batch, dims=ndims(x_batch)))
        true
    end

    single_std = rand(Float32, 2, 2)
    @test begin
        ds = DiagGaussian.(eachslice(mean_batch, dims=ndims(mean_batch)), Ref(single_std))
        entropies = DRiLDistributions.entropy.(ds)
        logpdfs = DRiLDistributions.logpdf.(ds, eachslice(x_batch, dims=ndims(x_batch)))
        true
    end
end
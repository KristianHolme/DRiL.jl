@testitem "TB logger converts and logs without error" begin
    using TensorBoardLogger
    using DRiL
    mktempdir() do dir
        raw = TBLogger(dir, tb_increment)
        lg = convert(DRiL.AbstractTrainingLogger, raw)
        DRiL.set_step!(lg, 1)
        DRiL.log_scalar!(lg, "x", 1.0)
        DRiL.write_hparams!(lg, Dict("lr" => 0.01, "gamma" => 0.99), ["env/ep_rew_mean"])
        DRiL.flush!(lg)
        DRiL.close!(lg)
        @test isdir(dir)
    end
end

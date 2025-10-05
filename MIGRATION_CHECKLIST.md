# Migration Checklist: Detailed File-by-File Changes

This checklist provides specific, actionable changes needed for each file in the migration from Zygote to Enzyme/Reactant and adding multi-device support.

## Legend
- 🟢 **Low complexity** - Simple changes, low risk
- 🟡 **Medium complexity** - Moderate refactoring, some risk
- 🔴 **High complexity** - Significant changes, careful testing required
- ⭐ **Critical path** - Must be done first

---

## Phase 1: Enzyme Migration

### 1. Dependencies

#### 🟢⭐ `Project.toml`
**Changes needed:**
- Add Enzyme dependency
- Update Lux version if needed (ensure Enzyme compatibility)

**Specific changes:**
```toml
[deps]
# Add this line
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

# Update if needed
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"

[compat]
# Add version constraint
Enzyme = "0.11, 0.12"
Lux = "1.12.4"  # Or latest with Enzyme support
```

#### 🟢 `test/Project.toml`
**Changes needed:**
- Keep Zygote for comparison tests (optional)
- Add Enzyme for testing

**Specific changes:**
```toml
[deps]
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
# Keep Zygote for now
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
```

---

### 2. Core Algorithm Files

#### 🟢⭐ `src/algorithms/ppo.jl`
**Line 62** - Update default AD type:
```julia
# Current:
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; 
    ad_type::Lux.Training.AbstractADType = AutoZygote(), 
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}

# Change to:
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; 
    ad_type::Lux.Training.AbstractADType = AutoEnzyme(), 
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}
```

**Line 162** - No changes needed (already uses ad_type parameter)

**Line 333** - Test `@ignore_derivatives` compatibility:
```julia
# Current code should work, but verify:
@ignore_derivatives maybe_normalize!(advantages, alg)
```

**Line 346-362** - Test `@ignore_derivatives` block:
```julia
# Current code should work, but verify:
stats = @ignore_derivatives begin
    # ... stats calculation
end
```

**Action items:**
- [ ] Change default AD type to AutoEnzyme()
- [ ] Run tests to verify compatibility
- [ ] Benchmark gradient computation speed
- [ ] Keep option to pass AutoZygote() as parameter for fallback

#### 🟢⭐ `src/algorithms/sac.jl`
**Line 396** - Update default AD type:
```julia
# Current:
function learn!(agent::SACAgent, replay_buffer::ReplayBuffer, env::AbstractParallelEnv, 
    alg::OffPolicyAlgorithm, max_steps::Int;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
    ad_type::Lux.Training.AbstractADType = AutoZygote())

# Change to:
function learn!(agent::SACAgent, replay_buffer::ReplayBuffer, env::AbstractParallelEnv, 
    alg::OffPolicyAlgorithm, max_steps::Int;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
    ad_type::Lux.Training.AbstractADType = AutoEnzyme())
```

**Lines 499, 526, 549** - No changes needed (already use ad_type)

**Line 61, 95** - Test `@ignore_derivatives` compatibility:
```julia
# Verify these work with Enzyme:
loss = -(log_ent_coef * @ignore_derivatives(log_probs_pi .+ target_entropy |> mean))
# and
target_q_values = @ignore_derivatives begin
    # ...
end
```

**Action items:**
- [ ] Change default AD type to AutoEnzyme()
- [ ] Verify three gradient computations (entropy, critic, actor)
- [ ] Test ignore_derivatives blocks
- [ ] Benchmark against Zygote

---

### 3. Type Stability Fixes

#### 🔴⭐ `src/policies.jl`
**Critical:** Multiple runtime dispatch issues must be fixed for Enzyme compatibility.

**Line 602** - Fix DiagGaussian dispatch:
```julia
# Current (with FIXME comment):
return DiagGaussian.(action_means_vec, Ref(log_std))

# Need to investigate type instability in DiagGaussian constructor
# May need to annotate or refactor DRiLDistributions
```

**Line 617** - Fix SquashedDiagGaussian dispatch:
```julia
# Current (with FIXME comment):
return SquashedDiagGaussian.(action_means_vec, Ref(log_std))

# Similar to above - investigate type instability
```

**Line 591** - Fix Q-critic value prediction:
```julia
# Current (with TODO comment):
values, critic_st = policy.critic_head(inputs, ps.critic_head, st.critic_head)

# May need function barrier or type annotation
```

**Lines 669-674** - Fix evaluate_actions dispatch:
```julia
# Current has multiple dispatch points:
new_action_means, st = get_actions_from_features(policy, actor_feats, ps, st)
values, st = get_values_from_features(policy, critic_feats, ps, st)
distributions = get_distributions(policy, new_action_means, ps.log_std)
```

**Action items:**
- [ ] Run `@code_warntype` on all policy forward passes
- [ ] Use JET.jl to identify all type instabilities
- [ ] Add type annotations where needed
- [ ] Consider function barriers for unstable sections
- [ ] Create type-stable versions of distribution constructors
- [ ] Test with Enzyme after each fix
- [ ] Benchmark to ensure no performance regression

**Testing approach:**
```julia
using JET
using DRiL

# Create test policy
policy = ContinuousActorCriticPolicy(Box(-1f0, 1f0, (4,)), Box(-1f0, 1f0, (2,)))
ps, st = Lux.setup(Random.default_rng(), policy)
obs = rand(Float32, 4, 10)

# Check type stability
@report_opt policy(obs, ps, st)
```

#### 🟡 `src/DRiLDistributions/diagGaussian.jl`
**Issue:** Called from policies.jl with potential type instability

**Action items:**
- [ ] Review constructor and ensure type stability
- [ ] Add explicit type parameters if needed
- [ ] Test with @code_warntype

#### 🟡 `src/DRiLDistributions/squashedDiagGaussian.jl`
**Issue:** Similar to DiagGaussian

**Action items:**
- [ ] Review constructor and ensure type stability
- [ ] Add explicit type parameters if needed
- [ ] Test with @code_warntype

---

### 4. Test Files

#### 🟢 `test/test_callbacks.jl`
**Lines 2, 57** - Update imports:
```julia
# Current:
using Zygote

# Change to (or add alongside):
using Enzyme
```

Update test calls:
```julia
# Current:
learn!(agent, env, alg, 512)

# Could explicitly test both:
learn!(agent, env, alg, 512; ad_type=AutoEnzyme())
learn!(agent, env, alg, 512; ad_type=AutoZygote())  # Fallback test
```

**Action items:**
- [ ] Update imports
- [ ] Add Enzyme-specific tests
- [ ] Keep Zygote tests for comparison

#### 🟢 `test/test_sac.jl`
**Lines 191, 245, 282, 321** - Update AD type in gradient computation tests:
```julia
# Current:
Lux.Training.compute_gradients(AutoZygote(), ...)

# Test both:
Lux.Training.compute_gradients(AutoEnzyme(), ...)
Lux.Training.compute_gradients(AutoZygote(), ...)  # For comparison
```

**Action items:**
- [ ] Update gradient computation tests
- [ ] Add Enzyme vs Zygote comparison tests
- [ ] Verify gradients match (within tolerance)

---

## Phase 2: Multi-Device Support

### 5. New Files to Create

#### 🟡⭐ `src/device.jl` (NEW FILE)
**Purpose:** Device abstraction layer

**Full implementation:**
```julia
# Abstract device type
abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end

struct CUDADevice <: AbstractDevice
    id::Int
end

struct ROCmDevice <: AbstractDevice
    id::Int
end

struct MetalDevice <: AbstractDevice end

# Device constructors with safety checks
function get_device(::Type{CPUDevice})
    return CPUDevice()
end

function get_device(::Type{CUDADevice}, id::Int=0)
    @assert isdefined(@__MODULE__, :CUDA) "CUDA extension not loaded"
    if !CUDA.functional()
        @warn "CUDA not functional, falling back to CPU"
        return CPUDevice()
    end
    return CUDADevice(id)
end

function get_device(::Type{ROCmDevice}, id::Int=0)
    @assert isdefined(@__MODULE__, :AMDGPU) "AMDGPU extension not loaded"
    if !AMDGPU.functional()
        @warn "AMDGPU not functional, falling back to CPU"
        return CPUDevice()
    end
    return ROCmDevice(id)
end

function get_device(::Type{MetalDevice})
    @assert isdefined(@__MODULE__, :Metal) "Metal extension not loaded"
    if !Metal.functional()
        @warn "Metal not functional, falling back to CPU"
        return CPUDevice()
    end
    return MetalDevice()
end

# Default device - CPU
default_device() = CPUDevice()

# Device array type helpers (implemented in extensions)
device_array_type(::CPUDevice) = Array
# device_array_type(::CUDADevice) = CuArray  # In extension
# device_array_type(::ROCmDevice) = ROCArray  # In extension
# device_array_type(::MetalDevice) = MtlArray  # In extension

# Device transfer functions (core implementations)
to_device(x::AbstractArray, ::CPUDevice) = Array(x)
to_device(x::Number, ::AbstractDevice) = x
to_device(x::Nothing, ::AbstractDevice) = nothing

# Recursive transfer for nested structures
function to_device(x::NamedTuple, device::AbstractDevice)
    NamedTuple{keys(x)}(to_device(v, device) for v in values(x))
end

function to_device(x::Tuple, device::AbstractDevice)
    Tuple(to_device(v, device) for v in x)
end

# ComponentArray support
function to_device(x::ComponentArray, device::AbstractDevice)
    data = to_device(getdata(x), device)
    return ComponentArray(data, getaxes(x))
end

# Synchronization
synchronize(::CPUDevice) = nothing

# Query functions
is_cpu(::CPUDevice) = true
is_cpu(::AbstractDevice) = false

is_gpu(::CPUDevice) = false
is_gpu(::AbstractDevice) = true

# String representation
Base.show(io::IO, ::CPUDevice) = print(io, "CPUDevice()")
Base.show(io::IO, d::CUDADevice) = print(io, "CUDADevice($(d.id))")
Base.show(io::IO, d::ROCmDevice) = print(io, "ROCmDevice($(d.id))")
Base.show(io::IO, ::MetalDevice) = print(io, "MetalDevice()")
```

**Action items:**
- [ ] Create file with device abstraction
- [ ] Implement CPU device (no dependencies)
- [ ] Add extension support for GPU backends
- [ ] Test device construction and queries
- [ ] Test to_device with nested structures

---

#### 🟢⭐ `ext/DRiLCUDAExt.jl` (NEW EXTENSION)
**Purpose:** CUDA-specific implementations

**Structure:**
```julia
module DRiLCUDAExt

using DRiL
using CUDA
import DRiL: to_device, synchronize, device_array_type

# Array transfer
to_device(x::AbstractArray, ::DRiL.CUDADevice) = CuArray(x)

# Synchronization
synchronize(::DRiL.CUDADevice) = CUDA.synchronize()

# Array type
device_array_type(::DRiL.CUDADevice) = CuArray

# Pinned memory for faster transfers
function DRiL.pin_memory!(buffer::DRiL.RolloutBuffer, ::DRiL.CUDADevice)
    # Only pin if on CPU
    CUDA.pin(buffer.observations)
    CUDA.pin(buffer.actions)
    CUDA.pin(buffer.rewards)
    CUDA.pin(buffer.advantages)
    CUDA.pin(buffer.returns)
    CUDA.pin(buffer.logprobs)
    CUDA.pin(buffer.values)
    return nothing
end

end # module
```

**Action items:**
- [ ] Create extension file
- [ ] Implement CUDA-specific functions
- [ ] Add pinned memory support
- [ ] Test CUDA transfers and synchronization

#### 🟢 `ext/DRiLAMDGPUExt.jl` (NEW EXTENSION)
**Purpose:** ROCm-specific implementations

**Similar structure to CUDA extension, using AMDGPU.jl**

**Action items:**
- [ ] Create extension file
- [ ] Implement AMDGPU-specific functions
- [ ] Test on AMD hardware (if available)

#### 🟢 `ext/DRiLMetalExt.jl` (NEW EXTENSION)
**Purpose:** Metal-specific implementations for Apple Silicon

**Similar structure, using Metal.jl**

**Action items:**
- [ ] Create extension file
- [ ] Implement Metal-specific functions
- [ ] Test on Apple Silicon (if available)

---

### 6. Core File Modifications for Devices

#### 🟡⭐ `src/DRiL.jl`
**Add device module:**
```julia
# After other includes, add:
include("device.jl")
export AbstractDevice, CPUDevice, CUDADevice, ROCmDevice, MetalDevice
export get_device, to_device, synchronize, default_device
```

**Action items:**
- [ ] Include device.jl
- [ ] Export device-related types and functions
- [ ] Update module documentation

#### 🔴⭐ `src/buffers.jl`
**Major refactoring needed for device support**

**Line 1-13** - Update RolloutBuffer structure:
```julia
# Current:
struct RolloutBuffer{T <: AbstractFloat, O, A} <: AbstractBuffer
    observations::Array{O}
    actions::Array{A}
    rewards::Vector{T}
    # ...
end

# Change to:
struct RolloutBuffer{T <: AbstractFloat, O, A, D <: AbstractDevice} <: AbstractBuffer
    observations::AbstractArray{O}
    actions::AbstractArray{A}
    rewards::AbstractVector{T}
    advantages::AbstractVector{T}
    returns::AbstractVector{T}
    logprobs::AbstractVector{T}
    values::AbstractVector{T}
    device::D
    gae_lambda::T
    gamma::T
    n_steps::Int
    n_envs::Int
end
```

**Line 18-30** - Update constructor:
```julia
function RolloutBuffer(
    observation_space::AbstractSpace, 
    action_space::AbstractSpace, 
    gae_lambda::T, 
    gamma::T, 
    n_steps::Int, 
    n_envs::Int;
    device::AbstractDevice = CPUDevice()
) where {T <: AbstractFloat}
    total_steps = n_steps * n_envs
    obs_eltype = eltype(observation_space)
    action_eltype = eltype(action_space)
    
    # Allocate on CPU first (for initial allocation)
    observations = Array{obs_eltype}(undef, size(observation_space)..., total_steps)
    actions = Array{action_eltype}(undef, size(action_space)..., total_steps)
    rewards = Vector{T}(undef, total_steps)
    advantages = Vector{T}(undef, total_steps)
    returns = Vector{T}(undef, total_steps)
    logprobs = Vector{T}(undef, total_steps)
    values = Vector{T}(undef, total_steps)
    
    # Transfer to device
    observations = to_device(observations, device)
    actions = to_device(actions, device)
    rewards = to_device(rewards, device)
    advantages = to_device(advantages, device)
    returns = to_device(returns, device)
    logprobs = to_device(logprobs, device)
    values = to_device(values, device)
    
    return RolloutBuffer{T, obs_eltype, action_eltype, typeof(device)}(
        observations, actions, rewards, advantages, returns, 
        logprobs, values, device, gae_lambda, gamma, n_steps, n_envs
    )
end
```

**Lines 121-200** - Update OffPolicyTrajectory:
```julia
# Similar changes: add device field and use AbstractArray
```

**Lines 201-250** - Update ReplayBuffer:
```julia
# Similar changes: add device field and constructor parameter
```

**Action items:**
- [ ] Update all buffer structures
- [ ] Add device field and parameter
- [ ] Change Array to AbstractArray throughout
- [ ] Update all constructors with device parameter
- [ ] Update reset! methods to work with device arrays
- [ ] Test on CPU and GPU

#### 🔴⭐ `src/agents.jl`
**Add device support to agents**

**Line 36-45** - Update ActorCriticAgent structure:
```julia
# Current:
mutable struct ActorCriticAgent{P <: AbstractActorCriticPolicy, R <: AbstractRNG} <: AbstractAgent
    policy::P
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing, TensorBoardLogger.TBLogger}
    verbose::Int
    rng::R
    stats::AgentStats
end

# Change to:
mutable struct ActorCriticAgent{P <: AbstractActorCriticPolicy, R <: AbstractRNG, D <: AbstractDevice} <: AbstractAgent
    policy::P
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing, TensorBoardLogger.TBLogger}
    verbose::Int
    rng::R
    stats::AgentStats
    device::D
end
```

**Line 68-82** - Update get_action_and_values:
```julia
function get_action_and_values(agent::ActorCriticAgent, observations::AbstractVector)
    policy = agent.policy
    train_state = agent.train_state
    ps = train_state.parameters
    st = train_state.states
    device = agent.device
    
    # Convert observations vector to batched matrix
    batched_obs = batch(observations, observation_space(policy))
    
    # Transfer to device if needed
    batched_obs_device = to_device(batched_obs, device)
    
    # Forward pass on device
    actions, values, logprobs, st = policy(batched_obs_device, ps, st)
    
    # Transfer back to CPU for environment interaction
    actions = to_device(actions, CPUDevice())
    values = to_device(values, CPUDevice())
    logprobs = to_device(logprobs, CPUDevice())
    
    @reset train_state.states = st
    agent.train_state = train_state
    return actions, values, logprobs
end
```

**Similar updates needed for:**
- predict_values (line 95-107)
- predict_actions (line 123-139)

**Action items:**
- [ ] Add device field to ActorCriticAgent
- [ ] Update constructor to accept device parameter
- [ ] Transfer parameters/states to device in constructor
- [ ] Update all inference methods with device transfers
- [ ] Add synchronization where needed

#### 🔴 `src/algorithms/sac.jl`
**Update SACAgent structure**

**Line 128-140** - Add device field:
```julia
mutable struct SACAgent{R <: AbstractRNG, D <: AbstractDevice} <: AbstractAgent
    policy::ContinuousActorCriticPolicy
    train_state::Lux.Training.TrainState
    Q_target_parameters::ComponentArray
    Q_target_states::NamedTuple
    ent_train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing, TensorBoardLogger.TBLogger}
    verbose::Int
    rng::R
    stats::AgentStats
    device::D
end
```

**Line 142-172** - Update constructor with device:
```julia
function SACAgent(
    policy::ContinuousActorCriticPolicy,
    alg::SAC;
    optimizer_type::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
    log_dir::Union{Nothing, String} = nothing,
    stats_window::Int = 100,
    rng::AbstractRNG = Random.default_rng(),
    verbose::Int = 1,
    device::AbstractDevice = CPUDevice()
)
    ps, st = Lux.setup(rng, policy)
    
    # Transfer to device
    ps = to_device(ps, device)
    st = to_device(st, device)
    
    # ... rest of setup
    
    Q_target_parameters = copy_critic_parameters(policy, ps)
    Q_target_states = copy_critic_states(policy, st)
    
    # ... entropy coefficient setup
    
    return SACAgent(
        policy, train_state, Q_target_parameters, Q_target_states,
        ent_train_state, optimizer_type, stats_window, logger, verbose, rng,
        AgentStats(0, 0), device
    )
end
```

**Line 202-226** - Update predict_actions:
```julia
function predict_actions(
    agent::SACAgent,
    observations::AbstractVector;
    deterministic::Bool = false,
    rng::AbstractRNG = agent.rng,
    raw::Bool = false
)
    train_state = agent.train_state
    policy = agent.policy
    ps = train_state.parameters
    st = train_state.states
    device = agent.device
    
    batched_obs = batch(observations, observation_space(policy))
    batched_obs_device = to_device(batched_obs, device)
    
    actions, st = predict_actions(policy, batched_obs_device, ps, st; deterministic, rng)
    
    # Transfer back to CPU
    actions = to_device(actions, CPUDevice())
    
    @reset train_state.states = st
    agent.train_state = train_state
    
    if raw
        return actions
    else
        alg = SAC()
        return process_action.(actions, Ref(action_space(policy)), Ref(alg))
    end
end
```

**Action items:**
- [ ] Add device field to SACAgent
- [ ] Update constructor with device parameter
- [ ] Update all prediction methods with device transfers
- [ ] Test SAC training on GPU

---

### 7. Training Loop Updates

#### 🔴⭐ `src/algorithms/ppo.jl`
**Major training loop refactoring**

**Line 69** - Pass device to buffer:
```julia
# Current:
roll_buffer = RolloutBuffer(
    observation_space(env), action_space(env),
    alg.gae_lambda, alg.gamma, n_steps, n_envs
)

# Change to:
roll_buffer = RolloutBuffer(
    observation_space(env), action_space(env),
    alg.gae_lambda, alg.gamma, n_steps, n_envs;
    device = agent.device
)
```

**Line 143-150** - Ensure data is on device:
```julia
# Data should already be on device from buffer, but verify:
data_loader = DataLoader(
    (
        roll_buffer.observations,  # Already on device
        roll_buffer.actions,
        roll_buffer.advantages,
        roll_buffer.returns,
        roll_buffer.logprobs,
        roll_buffer.values,
    ),
    batchsize = alg.batch_size, shuffle = true, parallel = true, rng = agent.rng
)
```

**Line 162** - Gradient computation on device:
```julia
# Should work as-is if data is on device
grads, loss_val, stats, train_state = Lux.Training.compute_gradients(
    ad_type, alg, batch_data, train_state
)
```

**Action items:**
- [ ] Pass device to RolloutBuffer
- [ ] Verify DataLoader works with device arrays
- [ ] Add synchronization after gradient updates if needed
- [ ] Test full training loop on GPU
- [ ] Profile for bottlenecks

#### 🔴⭐ `src/algorithms/sac.jl`
**Update SAC training loop**

**Line 384** - Pass device to buffer:
```julia
replay_buffer = ReplayBuffer(
    observation_space(env), action_space(env), alg.buffer_capacity;
    device = agent.device
)
```

**Lines 499, 526, 549** - Gradient computations should work as-is if data on device

**Action items:**
- [ ] Pass device to ReplayBuffer
- [ ] Verify get_data_loader works with device arrays
- [ ] Test three gradient computations on GPU
- [ ] Add synchronization where needed
- [ ] Profile performance

---

#### 🟡⭐ `src/buffers.jl`
**Update collection functions**

**Lines 74-115** - Update collect_trajectories (if used by PPO):
```julia
function collect_trajectories(
    agent::ActorCriticAgent, env::AbstractParallelEnv, alg::AbstractAlgorithm, n_steps::Int;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing
)
    # Trajectories stay on CPU (collected from environment)
    trajectories = Trajectory[]
    # ... rest of collection logic on CPU
    
    # Note: get_action_and_values handles device transfers internally
    actions, values, logprobs = get_action_and_values(agent, observations)
    
    # ... continue on CPU
end
```

**Lines 161-232** - Update collect_rollout!:
```julia
function collect_rollout!(
    buffer::RolloutBuffer, agent::ActorCriticAgent, alg::OnPolicyAlgorithm,
    env::AbstractParallelEnv;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing
)
    # Data collection on CPU, storage in buffer (which may be on GPU)
    device = agent.device
    
    # ... collection loop
    
    for step in 1:n_steps
        observations = observe(env)  # CPU
        
        # get_action_and_values handles device transfers
        actions, values, logprobs = get_action_and_values(agent, observations)
        
        # Store directly in buffer (handles device transfer if needed)
        buffer.observations[:, :, step] = to_device(observations, device)
        buffer.actions[:, step] = to_device(actions, device)
        # ...
    end
end
```

**Action items:**
- [ ] Review all collection functions
- [ ] Ensure CPU-GPU transfers are minimal
- [ ] Batch transfers where possible
- [ ] Add synchronization before computing advantages
- [ ] Test collection performance

---

### 8. Documentation Updates

#### 🟢 `README.md`
**Add device support section**

After line 90, add:
```markdown
### Device Support (GPU Training)

DRiL supports training on different devices:

```julia
using DRiL
using CUDA  # For NVIDIA GPUs

# Create agent on GPU
device = get_device(CUDADevice, 0)  # GPU 0
agent = ActorCriticAgent(
    policy,
    alg;
    device = device
)

# Training automatically happens on GPU
learn!(agent, env, ppo, max_steps)
```

**Supported devices:**
- CPU (default): `CPUDevice()`
- CUDA (NVIDIA): `CUDADevice(id)` - requires CUDA.jl
- ROCm (AMD): `ROCmDevice(id)` - requires AMDGPU.jl
- Metal (Apple): `MetalDevice()` - requires Metal.jl

**Performance tips:**
- Use GPU for larger networks (>1M parameters)
- Increase batch size on GPU for better utilization
- Environment interaction still happens on CPU
```

**Add AD backend section:**
```markdown
### Automatic Differentiation Backends

DRiL supports multiple AD backends:

```julia
# Enzyme (default, faster)
learn!(agent, env, alg, max_steps; ad_type=AutoEnzyme())

# Zygote (fallback)
learn!(agent, env, alg, max_steps; ad_type=AutoZygote())
```

Enzyme provides 1.5-3x speedup in gradient computation.
```

**Action items:**
- [ ] Add device support documentation
- [ ] Add AD backend documentation
- [ ] Add performance comparison section
- [ ] Add troubleshooting section
- [ ] Update installation instructions

---

## Testing Checklist

### Unit Tests to Add

#### 🟢 `test/test_device.jl` (NEW)
```julia
@testitem "Device abstraction" begin
    using DRiL
    
    @testset "CPU device" begin
        device = CPUDevice()
        arr = rand(Float32, 10, 10)
        arr_device = to_device(arr, device)
        @test arr_device isa Array
        @test arr_device == arr
    end
    
    @testset "Device transfers" begin
        device = CPUDevice()
        nt = (a = rand(Float32, 5), b = rand(Float32, 3, 3))
        nt_device = to_device(nt, device)
        @test nt_device.a == nt.a
        @test nt_device.b == nt.b
    end
end
```

#### 🟢 `test/test_enzyme.jl` (NEW)
```julia
@testitem "Enzyme gradient computation" begin
    using DRiL
    using Lux
    
    @testset "PPO gradients" begin
        # Test gradient computation with Enzyme
        # Compare with Zygote results (within tolerance)
    end
    
    @testset "SAC gradients" begin
        # Test all three gradient computations
    end
end
```

#### 🟡 `test/test_gpu_training.jl` (NEW)
```julia
@testitem "GPU training" tags=[:gpu] begin
    using DRiL
    using CUDA
    
    if CUDA.functional()
        @testset "PPO GPU training" begin
            device = get_device(CUDADevice, 0)
            # Create agent, train, verify results
        end
        
        @testset "SAC GPU training" begin
            # Similar test for SAC
        end
    else
        @warn "Skipping GPU tests - CUDA not functional"
    end
end
```

### Integration Tests to Add

#### 🟡 `test/test_end_to_end.jl`
```julia
@testitem "End-to-end with Enzyme" begin
    # Full training run with Enzyme
    # Verify convergence
end

@testitem "End-to-end CPU vs GPU" tags=[:gpu] begin
    # Train same problem on CPU and GPU
    # Verify results are similar (within tolerance)
end
```

---

## Performance Benchmarking

### Benchmarks to Add

#### `benchmarks/enzyme_vs_zygote.jl` (NEW)
```julia
using BenchmarkTools
using DRiL

# Benchmark gradient computation
@benchmark begin
    grads = Lux.Training.compute_gradients(AutoEnzyme(), ...)
end

@benchmark begin
    grads = Lux.Training.compute_gradients(AutoZygote(), ...)
end
```

#### `benchmarks/cpu_vs_gpu.jl` (NEW)
```julia
# Benchmark full training loops
# Vary network size and batch size
# Plot speedup curves
```

---

## Summary of File Changes

| File | Complexity | Critical | Enzyme | Multi-Device | Lines Changed |
|------|-----------|----------|--------|--------------|---------------|
| Project.toml | 🟢 | ⭐ | ✓ | ✓ | ~10 |
| src/DRiL.jl | 🟢 | ⭐ | | ✓ | ~5 |
| src/device.jl | 🟡 | ⭐ | | ✓ | ~200 (new) |
| src/algorithms/ppo.jl | 🟢 | ⭐ | ✓ | ✓ | ~20 |
| src/algorithms/sac.jl | 🟢 | ⭐ | ✓ | ✓ | ~30 |
| src/policies.jl | 🔴 | ⭐ | ✓ | | ~100 |
| src/agents.jl | 🔴 | ⭐ | | ✓ | ~80 |
| src/buffers.jl | 🔴 | ⭐ | | ✓ | ~150 |
| src/DRiLDistributions/*.jl | 🟡 | | ✓ | | ~50 |
| ext/DRiLCUDAExt.jl | 🟢 | | | ✓ | ~100 (new) |
| ext/DRiLAMDGPUExt.jl | 🟢 | | | ✓ | ~80 (new) |
| ext/DRiLMetalExt.jl | 🟢 | | | ✓ | ~80 (new) |
| test/*.jl | 🟢 | | ✓ | ✓ | ~100 |
| README.md | 🟢 | | ✓ | ✓ | ~50 |

**Total new lines:** ~750
**Total modified lines:** ~500
**Total new files:** 5

---

## Dependency Matrix

| Change | Depends On |
|--------|-----------|
| Enzyme in ppo.jl | Project.toml updated |
| Enzyme in sac.jl | Project.toml updated |
| Type stability fixes | None (can start immediately) |
| Device abstraction | None (can start immediately) |
| GPU extensions | Device abstraction complete |
| Buffer device support | Device abstraction complete |
| Agent device support | Device abstraction, Buffer changes |
| Training loop updates | All above complete |

## Estimated Effort

| Phase | Effort | Duration |
|-------|--------|----------|
| Enzyme migration | 2-3 weeks | Phase 1 |
| Type stability fixes | 2-3 weeks | Phase 1 (parallel) |
| Device abstraction | 1-2 weeks | Phase 2 |
| Buffer modifications | 1-2 weeks | Phase 2 |
| Agent modifications | 1-2 weeks | Phase 2 |
| Training loop updates | 2-3 weeks | Phase 2 |
| Testing & docs | 2-3 weeks | Continuous |
| **Total** | **11-18 weeks** | **3-5 months** |

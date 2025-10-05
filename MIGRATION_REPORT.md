# Migration Report: Zygote to Enzyme/Reactant and Multi-Device Support

## Executive Summary

This report analyzes the DRiL.jl codebase and outlines the changes required to:
1. Migrate automatic differentiation (AD) from Zygote to Enzyme and Reactant
2. Add support for different accelerator devices (CPU, CUDA GPUs, ROCm, Metal, etc.)

## Current State Analysis

### Automatic Differentiation

**Current Implementation:**
- Uses Zygote via Lux.Training.AbstractADType with `AutoZygote()`
- AD backend specified in two main locations:
  - `src/algorithms/ppo.jl:62` - PPO's `learn!()` function
  - `src/algorithms/sac.jl:396` - SAC's `learn!()` function
- Both use `Lux.Training.compute_gradients(ad_type, ...)` for gradient computation

**Zygote Characteristics:**
- Source-to-source reverse-mode AD
- Works on Julia's high-level IR
- Good for dynamic code, limited performance on very large models
- No explicit device support required

### Device/Hardware Support

**Current State:**
- No explicit device management
- All arrays are standard Julia `Array` types (CPU-bound)
- No CUDA, Metal, or ROCm support
- Data stays in CPU memory throughout training

**Key Locations:**
- Buffers (`src/buffers.jl`): Use `Array{T}` for all storage
- Policies (`src/policies.jl`): No device specification
- Training loops: No device transfers

---

## Target Backends

### Enzyme

**What is Enzyme:**
- LLVM-based automatic differentiation
- Forward and reverse mode AD at LLVM IR level
- Extremely high performance, especially for compiled code
- Better type stability and inference
- Native support for multi-threading and GPUs

**Advantages:**
- Faster gradient computation
- Better performance on loops and complex control flow
- Can differentiate through mutating operations
- Native GPU support

**Limitations:**
- More restrictive than Zygote (requires type-stable code)
- May not work with all Julia features
- Still maturing ecosystem

### Reactant

**What is Reactant:**
- XLA (Accelerated Linear Algebra) compilation for Julia
- Built on JAX's compiler infrastructure
- Supports multiple devices (CPU, GPU, TPU)
- Just-in-time (JIT) compilation and optimization

**Advantages:**
- Automatic device placement and optimization
- Cross-device compatibility (CUDA, ROCm, TPU)
- Advanced XLA optimizations
- Can fuse operations automatically

**Limitations:**
- Requires XLA-compatible operations
- More restrictive programming model
- Additional compilation overhead
- Currently experimental in Julia ecosystem

---

## Required Changes

### 1. AD Backend Migration

#### 1.1 Dependency Changes

**File: `Project.toml`**

Current dependencies:
```toml
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
```

**Required additions:**
```toml
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
Reactant = "3c362404-f566-11ee-1572-e11a3e718480"  # When stable
```

**Test dependencies (test/Project.toml):**
Remove or make optional:
```toml
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"  # Remove or keep for comparison tests
```

#### 1.2 Algorithm Changes

**File: `src/algorithms/ppo.jl`**

Current (line 62):
```julia
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; 
    ad_type::Lux.Training.AbstractADType = AutoZygote(), 
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}
```

**Proposed change:**
```julia
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; 
    ad_type::Lux.Training.AbstractADType = AutoEnzyme(),  # Change default
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}
```

Current (line 162):
```julia
grads, loss_val, stats, train_state = @timeit to "compute_gradients" 
    Lux.Training.compute_gradients(ad_type, alg, batch_data, train_state)
```

**No change needed** - Lux.Training.compute_gradients handles different AD backends.

**File: `src/algorithms/sac.jl`**

Current (line 396):
```julia
function learn!(agent::SACAgent, replay_buffer::ReplayBuffer, env::AbstractParallelEnv, 
    alg::OffPolicyAlgorithm, max_steps::Int;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
    ad_type::Lux.Training.AbstractADType = AutoZygote())
```

**Proposed change:**
```julia
function learn!(agent::SACAgent, replay_buffer::ReplayBuffer, env::AbstractParallelEnv, 
    alg::OffPolicyAlgorithm, max_steps::Int;
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing,
    ad_type::Lux.Training.AbstractADType = AutoEnzyme())
```

Multiple gradient computations (lines 499, 526, 549):
```julia
# Entropy coefficient gradient
Lux.Training.compute_gradients(ad_type, ...)

# Critic gradient  
Lux.Training.compute_gradients(ad_type, ...)

# Actor gradient
Lux.Training.compute_gradients(ad_type, ...)
```

**No change needed** - These already accept ad_type parameter.

#### 1.3 ChainRulesCore Usage

**File: `src/algorithms/ppo.jl` (lines 331-333, 346)**
```julia
@ignore_derivatives maybe_normalize!(advantages, alg)
# ...
stats = @ignore_derivatives begin
    # ...
end
```

**File: `src/algorithms/sac.jl` (lines 61, 95)**
```julia
loss = -(log_ent_coef * @ignore_derivatives(log_probs_pi .+ target_entropy |> mean))
# ...
target_q_values = @ignore_derivatives begin
    # ...
end
```

**Analysis:**
- `@ignore_derivatives` from ChainRulesCore is AD-backend agnostic
- Should work with Enzyme and Reactant
- **No changes required**, but may need testing for compatibility

**Potential issue:** Enzyme handles non-differentiable operations differently. May need to use:
```julia
# For Enzyme-specific code
Enzyme.Const(value)  # Mark as constant
```

#### 1.4 Code Compatibility Issues

**Type Stability Requirements:**
- Enzyme requires type-stable code
- Current code has some runtime dispatch (see `src/policies.jl` comments about "runtime dispatch")

**Files with potential issues:**
- `src/policies.jl:602` - "FIXME: runtime dispatch here in DiagGaussian"
- `src/policies.jl:617` - "FIXME: runtime dispatch here in SquashedDiagGaussian"
- `src/policies.jl:591` - "TODO: runtime dispatch"
- `src/policies.jl:669-674` - Multiple runtime dispatch comments

**Required fixes:**
1. Eliminate runtime dispatch in hot paths
2. Add type annotations where needed
3. Use function barriers appropriately
4. Test with `@code_warntype` and JET.jl

### 2. Multi-Device Support

#### 2.1 Device Abstraction Layer

**New file: `src/device.jl`**

Create device abstraction:
```julia
abstract type AbstractDevice end
struct CPUDevice <: AbstractDevice end
struct CUDADevice <: AbstractDevice 
    id::Int
end
struct ROCmDevice <: AbstractDevice
    id::Int
end
struct MetalDevice <: AbstractDevice end

# Device selection utilities
function get_device(::Type{CPUDevice})
    return CPUDevice()
end

function get_device(::Type{CUDADevice}, id::Int=0)
    if !CUDA.functional()
        @warn "CUDA not functional, falling back to CPU"
        return CPUDevice()
    end
    return CUDADevice(id)
end

# Array conversion
to_device(x::AbstractArray, ::CPUDevice) = Array(x)
to_device(x::AbstractArray, ::CUDADevice) = CuArray(x)
to_device(x::AbstractArray, ::ROCmDevice) = ROCArray(x)
to_device(x::AbstractArray, ::MetalDevice) = MtlArray(x)

# Recursive device transfer for nested structures
to_device(x::NamedTuple, device) = NamedTuple{keys(x)}(to_device(v, device) for v in values(x))
to_device(x::ComponentArray, device) = ComponentArray(to_device(getdata(x), device), getaxes(x))
```

#### 2.2 Dependency Additions

**File: `Project.toml`**

Add device backends:
```toml
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"  # For NVIDIA GPUs
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"  # For AMD GPUs  
Metal = "dde4c033-4e86-420c-a63e-0dd931031962"  # For Apple Silicon
LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"  # Lux CUDA integration

[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
Metal = "dde4c033-4e86-420c-a63e-0dd931763bda"

[extensions]
DRiLCUDAExt = "CUDA"
DRiLAMDGPUExt = "AMDGPU"
DRiLMetalExt = "Metal"
```

**Modern approach:** Use package extensions for optional GPU support.

#### 2.3 Buffer Changes

**File: `src/buffers.jl`**

Current:
```julia
struct RolloutBuffer{T <: AbstractFloat, O, A} <: AbstractBuffer
    observations::Array{O}
    actions::Array{A}
    rewards::Vector{T}
    # ...
end
```

**Proposed:**
```julia
struct RolloutBuffer{T <: AbstractFloat, O, A, D <: AbstractDevice} <: AbstractBuffer
    observations::AbstractArray{O}  # Changed from Array
    actions::AbstractArray{A}       # Changed from Array
    rewards::AbstractVector{T}      # Changed from Vector
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
    
    # Allocate on CPU first
    observations = Array{obs_eltype}(undef, size(observation_space)..., total_steps)
    actions = Array{action_eltype}(undef, size(action_space)..., total_steps)
    rewards = Vector{T}(undef, total_steps)
    # ... other allocations
    
    # Transfer to device
    observations = to_device(observations, device)
    actions = to_device(actions, device)
    rewards = to_device(rewards, device)
    # ... transfer others
    
    return RolloutBuffer{T, obs_eltype, action_eltype, typeof(device)}(
        observations, actions, rewards, advantages, returns, 
        logprobs, values, device, gae_lambda, gamma, n_steps, n_envs
    )
end
```

**Similar changes needed for:**
- `ReplayBuffer` (line 150+)
- `OffPolicyTrajectory` (line 121+)

#### 2.4 Agent and Policy Changes

**File: `src/agents.jl`**

Add device to agent:
```julia
mutable struct ActorCriticAgent{P <: AbstractActorCriticPolicy, R <: AbstractRNG, D <: AbstractDevice} <: AbstractAgent
    policy::P
    train_state::Lux.Training.TrainState
    optimizer_type::Type{<:Optimisers.AbstractRule}
    stats_window::Int
    logger::Union{Nothing, TensorBoardLogger.TBLogger}
    verbose::Int
    rng::R
    stats::AgentStats
    device::D  # Add device field
end
```

Update constructor:
```julia
function ActorCriticAgent(
    policy::AbstractActorCriticPolicy, alg::PPO;
    optimizer_type::Type{<:Optimisers.AbstractRule} = Optimisers.Adam,
    stats_window::Int = 100,
    verbose::Int = 1,
    log_dir::Union{Nothing, String} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    device::AbstractDevice = CPUDevice()  # Add device parameter
)
    optimizer = make_optimizer(optimizer_type, alg)
    ps, st = Lux.setup(rng, policy)
    
    # Transfer parameters and states to device
    ps = to_device(ps, device)
    st = to_device(st, device)
    
    # ... rest of setup
    
    train_state = Lux.Training.TrainState(policy, ps, st, optimizer)
    return ActorCriticAgent(
        policy, train_state, optimizer_type, stats_window,
        logger, verbose, rng, AgentStats(0, 0), device
    )
end
```

**File: `src/policies.jl`**

No structural changes needed, but ensure all forward passes work on device arrays.

#### 2.5 Training Loop Changes

**File: `src/algorithms/ppo.jl`**

Data collection (CPU) and training (GPU) separation:

```julia
function learn!(agent::ActorCriticAgent, env::AbstractParallelEnv, alg::PPO{T}, max_steps::Int; 
    ad_type::Lux.Training.AbstractADType = AutoEnzyme(),
    callbacks::Union{Vector{<:AbstractCallback}, Nothing} = nothing) where {T}
    
    device = agent.device
    n_steps = alg.n_steps
    n_envs = number_of_envs(env)
    
    # Buffer on device
    roll_buffer = RolloutBuffer(
        observation_space(env), action_space(env),
        alg.gae_lambda, alg.gamma, n_steps, n_envs;
        device = device
    )
    
    # ... training loop
    
    for i in 1:iterations
        # Collect rollout (happens on CPU with environment)
        fps, success = collect_rollout!(roll_buffer, agent, alg, env; callbacks = callbacks)
        
        # Transfer batch data to device if needed
        data_loader = DataLoader(
            (
                to_device(roll_buffer.observations, device),
                to_device(roll_buffer.actions, device),
                to_device(roll_buffer.advantages, device),
                to_device(roll_buffer.returns, device),
                to_device(roll_buffer.logprobs, device),
                to_device(roll_buffer.values, device),
            ),
            batchsize = alg.batch_size, shuffle = true, parallel = true, rng = agent.rng
        )
        
        # Training happens on device
        for epoch in 1:alg.epochs
            for (i_batch, batch_data) in enumerate(data_loader)
                # Gradients computed on device
                grads, loss_val, stats, train_state = 
                    Lux.Training.compute_gradients(ad_type, alg, batch_data, train_state)
                
                # ... rest of training
            end
        end
    end
end
```

**File: `src/algorithms/sac.jl`**

Similar changes for SAC's `learn!()` function (lines 388-601).

#### 2.6 Data Transfer Management

**New utilities needed:**

```julia
# src/device.jl

# Efficient batch transfer
function transfer_batch(obs_batch, action_batch, device)
    return (
        observations = to_device(obs_batch, device),
        actions = to_device(action_batch, device)
    )
end

# Pin memory for faster transfers (CUDA)
function pin_memory!(buffer::RolloutBuffer, ::CUDADevice)
    # Pin CPU arrays for faster GPU transfer
    CUDA.pin(buffer.observations)
    CUDA.pin(buffer.actions)
    # ... pin other arrays
end

# Synchronization
synchronize(::CPUDevice) = nothing
synchronize(::CUDADevice) = CUDA.synchronize()
synchronize(::ROCmDevice) = AMDGPU.synchronize()
```

#### 2.7 Environment Interaction

**Critical consideration:** Environments run on CPU, models on GPU.

Strategy:
1. Keep environment observations on CPU
2. Transfer batches to GPU for inference
3. Transfer actions back to CPU for environment
4. Accumulate trajectories on CPU
5. Transfer full buffers to GPU for training

**File: `src/buffers.jl` - Modify collection:**

```julia
function collect_rollout!(
    buffer::RolloutBuffer, 
    agent::ActorCriticAgent, 
    alg::OnPolicyAlgorithm,
    env::AbstractParallelEnv;
    callbacks = nothing
)
    device = agent.device
    # ... setup
    
    for step in 1:n_steps
        # Observations from environment (CPU)
        observations = observe(env)
        
        # Transfer to device for inference
        obs_batch = batch(observations, observation_space(policy))
        obs_batch_device = to_device(obs_batch, device)
        
        # Get actions on device
        actions_device, values_device, logprobs_device = 
            get_action_and_values_device(agent, obs_batch_device)
        
        # Transfer actions back to CPU for environment
        actions = to_device(actions_device, CPUDevice())
        
        # Process and act in environment (CPU)
        processed_actions = process_action.(actions, Ref(action_space), Ref(alg))
        rewards, terminateds, truncateds, infos = act!(env, processed_actions)
        
        # Store in buffer (already on device)
        # ... storage logic
    end
end
```

---

## Migration Strategy

### Phase 1: Enzyme Migration (Recommended First Step)

**Rationale:** Enzyme is more mature and requires fewer code changes.

**Steps:**

1. **Add dependencies** (Week 1)
   - Add Enzyme to Project.toml
   - Test Lux compatibility with Enzyme

2. **Fix type stability issues** (Week 2-3)
   - Run with `@code_warntype` on hot paths
   - Eliminate runtime dispatch in policies.jl
   - Add type annotations where needed
   - Test with JET.jl

3. **Update default AD backend** (Week 3)
   - Change `AutoZygote()` to `AutoEnzyme()` in learn! functions
   - Keep Zygote as fallback option via parameter

4. **Test and benchmark** (Week 4)
   - Run full test suite
   - Benchmark against Zygote
   - Fix any Enzyme-specific issues

5. **Documentation** (Week 4)
   - Update README with AD backend options
   - Document performance differences

### Phase 2: Multi-Device Support (After Enzyme)

**Rationale:** Device support builds on stable AD backend.

**Steps:**

1. **Device abstraction** (Week 5-6)
   - Implement src/device.jl
   - Add device parameter to core structures
   - Create package extensions for GPU backends

2. **Buffer modifications** (Week 7-8)
   - Update RolloutBuffer, ReplayBuffer
   - Add device transfer utilities
   - Implement efficient CPU-GPU transfers

3. **Agent and policy updates** (Week 9-10)
   - Add device field to agents
   - Update constructors with device parameter
   - Ensure all operations are device-agnostic

4. **Training loop updates** (Week 11-12)
   - Modify collect_rollout! for CPU-GPU split
   - Update learn! functions with device transfers
   - Add synchronization points

5. **Testing and optimization** (Week 13-14)
   - Test on CUDA, ROCm, Metal
   - Optimize transfer patterns
   - Benchmark CPU vs GPU training
   - Profile for bottlenecks

6. **Documentation and examples** (Week 15)
   - Add GPU training examples
   - Document device selection
   - Performance guidelines

### Phase 3: Reactant Integration (Optional/Future)

**Rationale:** Reactant is more experimental, consider after stable multi-device support.

**Steps:**

1. **Feasibility study** (Week 16)
   - Test Reactant with Lux
   - Identify compatibility issues
   - Evaluate performance benefits

2. **Code adaptation** (Week 17-20)
   - Convert operations to XLA-compatible forms
   - Handle compilation overhead
   - Test across devices

3. **Integration** (Week 21-22)
   - Add AutoReactant() option
   - Benchmark vs Enzyme
   - Document use cases

---

## Compatibility Matrix

| Feature | Zygote | Enzyme | Reactant |
|---------|--------|--------|----------|
| Type stability required | No | Yes | Yes |
| Mutation support | Limited | Yes | Limited |
| GPU support | Via arrays | Native | Native |
| Multi-device | Via arrays | Via arrays | Automatic |
| Performance | Good | Excellent | Excellent |
| Maturity | Stable | Maturing | Experimental |
| Lux integration | Excellent | Good | Developing |

---

## Risk Assessment

### High Risk Items

1. **Type stability in policies.jl**
   - Multiple runtime dispatch locations
   - May require significant refactoring
   - Could break existing API

2. **ChainRulesCore compatibility**
   - `@ignore_derivatives` behavior may differ
   - Needs thorough testing with Enzyme

3. **Device transfer overhead**
   - CPU-GPU transfers can be slow
   - Need efficient batching strategy
   - May impact training speed

### Medium Risk Items

1. **Buffer memory management**
   - Large buffers on GPU may exhaust memory
   - Need fallback strategies
   - Pinned memory management

2. **Reactant compatibility**
   - Very experimental
   - May not support all operations
   - Compilation overhead

3. **Testing coverage**
   - Need multi-device test infrastructure
   - CI/CD complexity increases

### Low Risk Items

1. **Basic Enzyme migration**
   - Lux already supports Enzyme
   - Well-documented API
   - Straightforward parameter change

2. **Package extensions**
   - Modern Julia feature
   - Clean optional dependencies

---

## Testing Strategy

### Unit Tests

1. **AD backend tests**
   ```julia
   @testset "Enzyme gradients" begin
       # Test gradient computation with Enzyme
       # Compare with Zygote results
   end
   ```

2. **Device transfer tests**
   ```julia
   @testset "Device transfers" begin
       # Test CPU -> GPU -> CPU roundtrip
       # Verify data integrity
   end
   ```

3. **Type stability tests**
   ```julia
   @testset "Type stability" begin
       # Use JET.jl to check inference
   end
   ```

### Integration Tests

1. **Full training runs**
   - Test PPO with Enzyme + CPU
   - Test PPO with Enzyme + CUDA
   - Test SAC with Enzyme + CUDA

2. **Performance benchmarks**
   - Compare training speeds
   - Measure GPU utilization
   - Profile memory usage

### Compatibility Tests

1. **Multi-backend tests**
   - Ensure Zygote still works as fallback
   - Test device-agnostic code paths

---

## Performance Expectations

### Enzyme vs Zygote

- **Expected speedup:** 1.5-3x for gradient computation
- **Compilation time:** Slightly longer first run
- **Memory usage:** Similar or slightly better

### GPU vs CPU

- **Expected speedup:** 3-10x depending on:
  - Model size (larger = better speedup)
  - Batch size (larger = better GPU utilization)
  - Rollout collection overhead (CPU-bound)

### Bottlenecks

1. **Environment interaction** (CPU-bound)
   - Cannot be accelerated on GPU
   - Dominates small models
   - Use parallel environments to mitigate

2. **Data transfers** (bandwidth-limited)
   - Minimize transfer frequency
   - Use pinned memory
   - Batch transfers efficiently

3. **Small batch sizes** (GPU underutilization)
   - Increase batch size for GPU training
   - May need to adjust hyperparameters

---

## Recommendations

### Immediate Actions

1. **Start with Enzyme migration**
   - Low risk, high reward
   - Validates code quality (type stability)
   - Foundation for device support

2. **Fix type stability issues**
   - Required for Enzyme
   - Benefits Zygote too
   - Improves overall code quality

3. **Design device abstraction**
   - Plan API before implementation
   - Consider future extensibility
   - Get community feedback

### Medium-Term Actions

1. **Implement CUDA support first**
   - Most common GPU backend
   - Best ecosystem support
   - Easier to test

2. **Add comprehensive benchmarks**
   - Track performance regressions
   - Validate device speedups
   - Guide optimization efforts

3. **Update documentation**
   - Device selection guide
   - Performance tuning tips
   - Migration guide for users

### Long-Term Actions

1. **Evaluate Reactant**
   - When more mature
   - For cross-platform support
   - Advanced optimization needs

2. **Distributed training**
   - Multi-GPU support
   - Across-node parallelism
   - Builds on device abstraction

3. **Automatic device selection**
   - Heuristics for CPU vs GPU
   - Memory-aware scheduling
   - Fallback strategies

---

## Open Questions

1. **How to handle mixed-device scenarios?**
   - Some agents on GPU, some on CPU
   - Multi-GPU load balancing

2. **Should we support half-precision (Float16) training?**
   - Significant speedup on modern GPUs
   - May impact training stability

3. **How to handle device-specific optimizations?**
   - Custom kernels for critical operations
   - Trade-off with code maintainability

4. **What's the minimum GPU memory requirement?**
   - Document requirements per model size
   - Provide memory-efficient modes

5. **Should we support TPUs via Reactant?**
   - Cloud training scenarios
   - Access and testing challenges

---

## Conclusion

**Migration from Zygote to Enzyme is feasible and recommended** with estimated timeline of 4 weeks. The main challenge is ensuring type stability in the policy code.

**Adding multi-device support is a larger undertaking** (10-15 weeks) but follows a clear path:
1. Device abstraction layer
2. Buffer modifications  
3. Agent/policy updates
4. Training loop changes
5. Testing and optimization

**Reactant integration should be deferred** until the ecosystem matures and Enzyme + device support is stable.

The biggest architectural decision is the **CPU-GPU split for environment interaction**, which requires careful design to minimize transfer overhead while maximizing GPU utilization during training.

**Expected benefits:**
- Enzyme: 1.5-3x faster gradient computation
- GPU: 3-10x faster training for large models
- Better type stability and code quality
- Future-proof for advanced accelerators

**Key risks to manage:**
- Type stability refactoring
- Device transfer overhead
- Increased testing complexity
- GPU memory constraints

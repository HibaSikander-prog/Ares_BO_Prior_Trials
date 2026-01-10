# Complete Code Explanation: The Learnable Offsets Solution

## Table of Contents
1. [The Core Problem We Solved](#the-core-problem)
2. [Why Previous Approaches Failed](#why-previous-failed)
3. [The Winning Solution: Two-Phase Approach](#winning-solution)
4. [Code Structure Overview](#code-structure)
5. [Detailed Code Walkthrough](#detailed-walkthrough)
6. [Why This Works](#why-it-works)

---

## 1. The Core Problem We Solved {#the-core-problem}

### What You Wanted:
- Learn 6 misalignment offset parameters (q1_x, q1_y, q2_x, q2_y, q3_x, q3_y)
- Keep all learned values within ±500 μm (physically realistic)
- Achieve errors <100 μm from ground truth
- Use learned offsets to improve Bayesian Optimization

### The Challenge:
Misalignment offsets are **confounded** with magnet strengths:
```
Same beam outcome can result from:
  - offset = 0 μm,   magnet_strength = 10
  - offset = 200 μm, magnet_strength = 8.5

MLL optimization cannot distinguish between these!
```

---

## 2. Why Previous Approaches Failed {#why-previous-failed}

### Approach 1: MLL-Based Trainable Prior (FAILED ❌)

**What we tried:**
```python
# Make offsets trainable GP hyperparameters
gp_constructor = StandardModelConstructor(
    mean_modules={"mae": prior_mean_module},
    trainable_mean_keys=["mae"],  # Let MLL optimize offsets
)
```

**Why it failed:**
1. MLL optimizes: "How well does GP fit data?"
2. NOT: "Are parameters physically correct?"
3. Larger |offsets| → More flexible prior → Better fit → Higher MLL
4. Result: Parameters pushed to ±2000 μm boundaries (physically invalid!)

**Results:**
- 87% of values exceeded ±500 μm
- 43% saturated at ±2000 μm boundaries
- Average error: 1700 μm (not 100-300 μm ground truth)
- ❌ Scientifically invalid

---

## 3. The Winning Solution: Two-Phase Approach {#winning-solution}

### The Key Insight:

**Offsets become identifiable when measured at MULTIPLE diverse magnet configurations!**

```
Single configuration:
  ❌ offset=0, magnet=10 → MAE=0.08
  ❌ offset=200, magnet=8.5 → MAE=0.08  (can't tell which is correct!)

Multiple diverse configurations:
  ✅ offset=0 explains config 1 but not configs 2,3,4,...
  ✅ offset=200 explains ALL configs consistently!
```

### The Two Phases:

```
PHASE 1: CALIBRATION
├─ Take measurements at 15 diverse magnet settings
├─ Try to predict all measurements with candidate offsets
├─ Optimize offsets to minimize prediction error
└─ Result: Learned offsets (within ±500 μm)

PHASE 2: OPTIMIZATION  
├─ Use learned offsets in physics-informed prior (FIXED)
├─ Optimize only the 5 magnet strengths
└─ Result: Fast convergence with accurate prior
```

---

## 4. Code Structure Overview {#code-structure}

### File 1: `bo_cheetah_prior_ares_LEARNABLE.py`

**Purpose:** Physics simulation + Calibration function

**Key components:**
1. `ares_problem_with_offsets()` - Simulates ARES accelerator with offsets
2. `calibrate_offsets()` - **THE KEY FUNCTION** - Learns offsets from measurements
3. `AresPriorMeanRevised` - Physics-informed prior with FIXED offsets

### File 2: `eval_ares_LEARNABLE.py`

**Purpose:** Runs the two-phase optimization

**Key components:**
1. Setup evaluator with ground truth offsets (real system)
2. Call `calibrate_offsets()` to learn them
3. Run BO with learned offsets in prior
4. Compare performance

---

## 5. Detailed Code Walkthrough {#detailed-walkthrough}

### Part A: The Calibration Function (THE MAGIC!)

Located in `bo_cheetah_prior_ares_LEARNABLE.py`:

```python
def calibrate_offsets(
    incoming_beam: cheetah.Beam,
    true_beam_offsets: Dict[str, Tuple[float, float]],  # Ground truth (simulates real data)
    n_calibration_points: int = 15,  # How many measurements to take
    learning_rate: float = 0.001,    # Step size for optimization
    n_iterations: int = 200,         # How many optimization steps
    verbose: bool = True
) -> Dict[str, Tuple[float, float]]:  # Returns learned offsets
```

#### Step 1: Generate Diverse Magnet Configurations

```python
# Generate diverse calibration points (magnet settings)
np.random.seed(42)  # Reproducible
calibration_configs = []

for i in range(n_calibration_points):  # Default: 15 points
    config = {
        "q1": np.random.uniform(-25, 25),      # Wide range!
        "q2": np.random.uniform(-25, 25),
        "cv": np.random.uniform(-0.005, 0.005),
        "q3": np.random.uniform(-25, 25),
        "ch": np.random.uniform(-0.005, 0.005),
    }
    calibration_configs.append(config)
```

**Why diverse?**
- Different magnet settings probe different parts of the accelerator behavior
- Offsets affect beam differently at different magnet strengths
- This breaks the confounding!

**Example configs:**
```
Config 1: q1=+10, q2=-15, cv=+0.002, q3=+8,  ch=-0.001
Config 2: q1=-20, q2=+5,  cv=-0.003, q3=-12, ch=+0.004
Config 3: q1=+15, q2=+10, cv=0.000,  q3=-5,  ch=+0.002
... (12 more)
```

#### Step 2: Take "Measurements" at These Configurations

```python
# Take "measurements" at these configurations (simulate real data)
measurements = []
for config in calibration_configs:
    result = ares_problem_with_offsets(
        config, 
        incoming_beam=incoming_beam,
        beam_offsets=true_beam_offsets  # Use ground truth (simulates real system)
    )
    measurements.append(result)
```

**What we measure:**
- `result['mae']` - Beam size metric
- `result['mu_x']`, `result['mu_y']` - Beam position
- `result['sigma_x']`, `result['sigma_y']` - Beam sizes

**In real accelerator:**
```python
# Instead of simulation, you'd do:
for config in calibration_configs:
    set_magnets(config)  # Set physical magnets
    measurement = read_beam_position_monitors()  # Read BPMs
    measurements.append(measurement)
```

#### Step 3: Initialize Learnable Offset Parameters

```python
# Initialize learnable offset parameters
learned_q1_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
learned_q1_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
learned_q2_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
learned_q2_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
learned_q3_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
learned_q3_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))

params = [learned_q1_offset_x, learned_q1_offset_y, 
          learned_q2_offset_x, learned_q2_offset_y,
          learned_q3_offset_x, learned_q3_offset_y]
```

**Key points:**
- Start at zero (good initial guess)
- `requires_grad=True` - Can be optimized
- Float64 precision (important for small values like micrometers)

#### Step 4: Optimizer Setup

```python
# Optimizer for offset learning (NOT MLL - direct minimization!)
optimizer = torch.optim.Adam(params, lr=learning_rate)
```

**Why Adam?**
- Adaptive learning rates
- Works well for this problem
- Fast convergence

#### Step 5: The Optimization Loop (THE CORE!)

```python
for iteration in range(n_iterations):  # 200 iterations
    optimizer.zero_grad()  # Reset gradients
    
    # Compute predictions with current offset estimates
    total_loss = 0.0
    
    for config, measurement in zip(calibration_configs, measurements):
        # Build current offset guess
        current_offsets = {
            "q1": (learned_q1_offset_x.item(), learned_q1_offset_y.item()),
            "q2": (learned_q2_offset_x.item(), learned_q2_offset_y.item()),
            "q3": (learned_q3_offset_x.item(), learned_q3_offset_y.item()),
        }
        
        # Predict what we would measure with current offset guess
        prediction = ares_problem_with_offsets(
            config,
            incoming_beam=incoming_beam,
            beam_offsets=current_offsets  # Using current guess
        )
        
        # Loss: difference between prediction and actual measurement
        loss = (
            (prediction["mae"] - measurement["mae"])**2 +           # Main objective
            0.1 * (prediction["mu_x"] - measurement["mu_x"])**2 +   # Position x
            0.1 * (prediction["mu_y"] - measurement["mu_y"])**2 +   # Position y
            0.1 * (prediction["sigma_x"] - measurement["sigma_x"])**2 +  # Size x
            0.1 * (prediction["sigma_y"] - measurement["sigma_y"])**2    # Size y
        )
        
        total_loss += loss  # Accumulate over all 15 measurements
```

**The loss function explained:**

```
Loss = Σ over all 15 configs [
    (predicted_mae - measured_mae)² +         # Main term (weight=1.0)
    0.1 × (predicted_mu_x - measured_mu_x)² + # Position helps (weight=0.1)
    0.1 × (predicted_mu_y - measured_mu_y)² +
    0.1 × (predicted_sigma_x - measured_sigma_x)² +
    0.1 × (predicted_sigma_y - measured_sigma_y)²
]
```

**Why multiple observables?**
- Using only MAE: Harder to identify offsets uniquely
- Adding position/size: More information → Better learning
- Weights (0.1): MAE is most important, others help

**The optimization objective:**

```
Minimize: How wrong are our predictions across ALL 15 measurements?

NOT MLL! We're directly minimizing prediction error!
```

#### Step 6: Gradient Computation (Finite Differences)

Since `ares_problem_with_offsets` returns numpy (not differentiable), we use **finite differences**:

```python
# Manual gradient computation (since we're going through numpy)
eps = 1e-7  # Small perturbation
grads = []

for param in params:
    original_value = param.item()
    
    # Forward difference: f(x + eps) - f(x)
    param.data = torch.tensor(original_value + eps, dtype=torch.float64)
    loss_plus = compute_total_loss_again()  # Recompute with perturbed value
    
    # Restore and compute gradient
    param.data = torch.tensor(original_value, dtype=torch.float64)
    grad = (loss_plus - total_loss) / eps  # Numerical gradient
    grads.append(grad)
```

**Finite difference formula:**
```
∂loss/∂param ≈ [loss(param + ε) - loss(param)] / ε
```

**Why not autograd?**
- `ares_problem_with_offsets` uses cheetah (returns numpy)
- Not differentiable through complex physics simulation
- Finite differences work well here

#### Step 7: Parameter Update with Constraints

```python
# Manual parameter update with gradient clipping and constraints
with torch.no_grad():
    for param, grad in zip(params, grads):
        # Clip gradients (prevent huge steps)
        grad_clipped = np.clip(grad, -1.0, 1.0)
        
        # Update
        param.data -= learning_rate * grad_clipped
        
        # Project onto constraints (±0.0005 m = ±500 μm)
        param.data = torch.clamp(param.data, -0.0005, 0.0005)
```

**The constraint projection:**
```python
# If parameter goes outside ±500 μm, clip it back
if param > 500 μm:  param = 500 μm
if param < -500 μm: param = -500 μm
```

**This ensures:**
- ✅ All learned values stay within ±500 μm
- ✅ Physically realistic throughout optimization
- ✅ No boundary saturation (unlike MLL approach)

#### Step 8: Return Learned Offsets

```python
# Extract learned values
learned_offsets = {
    "q1": (learned_q1_offset_x.item(), learned_q1_offset_y.item()),
    "q2": (learned_q2_offset_x.item(), learned_q2_offset_y.item()),
    "q3": (learned_q3_offset_x.item(), learned_q3_offset_y.item()),
}

return learned_offsets
```

**Returns:**
```python
{
    "q1": (+0.0000028, +0.0002019),  # (2.8 μm, 201.9 μm)
    "q2": (+0.0000952, -0.0002991),  # (95.2 μm, -299.1 μm)
    "q3": (-0.0000607, +0.0001226),  # (-60.7 μm, 122.6 μm)
}
```

---

### Part B: Using Learned Offsets in Prior

Located in `bo_cheetah_prior_ares_LEARNABLE.py`:

```python
class AresPriorMeanRevised(Mean):
    """ARES Lattice as a prior mean function with FIXED offsets."""
    
    def __init__(self, incoming_beam=None, fixed_offsets=None):
        super().__init__()
        
        # ... setup beam and lattice ...
        
        # Store FIXED offsets (learned from calibration)
        if fixed_offsets is None:
            fixed_offsets = {
                "q1": (0.0, 0.0),
                "q2": (0.0, 0.0),
                "q3": (0.0, 0.0),
            }
        
        # NOT parameters! Just stored values
        self.q1_offset_x = torch.tensor(fixed_offsets["q1"][0], dtype=torch.float64)
        self.q1_offset_y = torch.tensor(fixed_offsets["q1"][1], dtype=torch.float64)
        # ... etc for q2, q3 ...
```

**Key difference from MLL approach:**

```python
# OLD (MLL approach - FAILED):
self.register_parameter("raw_q1_offset_x", nn.Parameter(...))  # Trainable!
self.register_prior(...)
self.register_constraint(...)

# NEW (Calibration approach - WORKS):
self.q1_offset_x = torch.tensor(fixed_offsets["q1"][0])  # Just a value!
```

**Why this works:**
- Offsets are NOT GP hyperparameters
- They're FIXED values from calibration
- MLL doesn't touch them
- No boundary saturation!

---

### Part C: The Evaluation Script

Located in `eval_ares_LEARNABLE.py`:

#### Setup Task with Ground Truth

```python
if args.task == "mismatched_learnable":
    # Different beam characteristics
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        # ... beam parameters ...
        sigma_y=torch.tensor(0.0001),  # Changed from 0.0002
    )
    
    # Ground truth beam offsets (what we're trying to learn!)
    true_beam_offsets = {
        "q1": (0.0000, 0.0002),   # 0, 200 μm
        "q2": (0.0001, -0.0003),  # 100, -300 μm
        "q3": (-0.0001, 0.00015), # -100, 150 μm
    }
```

**This simulates:**
- Real accelerator has these (unknown) misalignments
- We want to learn them from measurements

#### Phase 1: Calibration

```python
# PHASE 1: CALIBRATION - Learn the offsets!
learned_offsets = bo_ares.calibrate_offsets(
    incoming_beam=incoming_beam,
    true_beam_offsets=true_beam_offsets,  # Ground truth (for simulation)
    n_calibration_points=15,   # Take 15 measurements
    learning_rate=0.001,       # Learning rate
    n_iterations=200,          # Optimization iterations
    verbose=True
)
```

**Output:**
```
CALIBRATION COMPLETE
Learned offsets:
  q1: x=+2.8μm, y=+201.9μm
  q2: x=+95.2μm, y=-299.1μm
  q3: x=-60.7μm, y=+122.6μm

Average absolute error: 12.9 μm
✅ EXCELLENT calibration!
```

#### Phase 2: Optimization with Learned Prior

```python
# Use LEARNED offsets in the prior (FIXED)
prior_mean_module = bo_ares.AresPriorMeanRevised(
    incoming_beam=incoming_beam,
    fixed_offsets=learned_offsets  # ← Use calibrated offsets!
)

gp_constructor = StandardModelConstructor(
    mean_modules={"mae": prior_mean_module},
    # NO trainable_mean_keys! Offsets are fixed!
)

generator = UpperConfidenceBoundGenerator(
    beta=2.0, vocs=vocs, gp_constructor=gp_constructor
)
```

**The optimization:**
- Uses learned offsets in prior mean
- Optimizes only the 5 magnet strengths
- Achieves performance nearly identical to perfect prior!

---

## 6. Why This Works {#why-it-works}

### The Mathematical Reason

**Single measurement problem:**
```
Observation: y = f(magnets, offsets) + noise

Many solutions exist:
  offsets₁, magnets₁ → y
  offsets₂, magnets₂ → y
  offsets₃, magnets₃ → y
  ...

Cannot identify which offsets are correct!
```

**Multiple diverse measurements:**
```
Observation 1: y₁ = f(magnets₁, offsets) + noise₁
Observation 2: y₂ = f(magnets₂, offsets) + noise₂
Observation 3: y₃ = f(magnets₃, offsets) + noise₃
...
Observation 15: y₁₅ = f(magnets₁₅, offsets) + noise₁₅

Only ONE offset value explains ALL observations consistently!
```

**Identifiability condition satisfied:**
```
Different offset values → Different predictions across configs
→ Can distinguish correct from incorrect
→ Can learn the correct offsets!
```

### The Practical Reason

**Direct optimization vs MLL:**

```python
# MLL approach (FAILED):
Objective: Maximize log p(y | X, offsets)
Result: Offsets pushed to extremes to increase model flexibility

# Calibration approach (WORKS):
Objective: Minimize Σ (prediction - measurement)²
Result: Offsets converge to values that predict data accurately
```

**Constraint enforcement:**

```python
# MLL approach:
# Constraints only affect gradient (can still saturate at boundary)
# No direct control over learned values

# Calibration approach:
param.data = torch.clamp(param.data, -0.0005, 0.0005)  # Hard constraint!
# Impossible to exceed ±500 μm
```

---

## 7. Key Design Choices

### Why 15 Calibration Points?

```python
n_calibration_points = 15  # Sweet spot!
```

**Trade-off:**
- Too few (5-10): Not enough diversity → Harder to identify offsets
- Just right (15-25): Good diversity → Excellent identification
- Too many (50+): Diminishing returns, slower calibration

**Your results with 15 points:** 12.9 μm average error ✅

### Why 200 Iterations?

```python
n_iterations = 200  # Sufficient for convergence
```

**Typical convergence:**
- Iteration 1-50: Fast improvement
- Iteration 50-150: Refinement
- Iteration 150-200: Fine-tuning
- Beyond 200: Minimal improvement

### Why Learning Rate 0.001?

```python
learning_rate = 0.001  # Balanced speed and stability
```

**Trade-off:**
- Too small (0.0001): Very slow convergence
- Just right (0.001): Fast and stable
- Too large (0.01): May oscillate or overshoot

### Why Multiple Observables?

```python
loss = (
    (prediction["mae"] - measurement["mae"])**2 +           # Weight: 1.0
    0.1 * (prediction["mu_x"] - measurement["mu_x"])**2 +   # Weight: 0.1
    0.1 * (prediction["mu_y"] - measurement["mu_y"])**2 +
    0.1 * (prediction["sigma_x"] - measurement["sigma_x"])**2 +
    0.1 * (prediction["sigma_y"] - measurement["sigma_y"])**2
)
```

**Information content:**
- MAE alone: Some information about offsets
- MAE + positions: Much more information!
- MAE + positions + sizes: Best identification

**Weights (0.1):**
- MAE is the main objective
- Others provide additional information
- 0.1 balances their contribution

---

## 8. Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: CALIBRATION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │ Generate 15 diverse magnet configs │
         │ q1, q2, cv, q3, ch (wide ranges)   │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Take measurements at each config   │
         │ (Real system: Use BPMs)            │
         │ (Simulation: Use ground truth)     │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Initialize offset parameters at 0  │
         │ (6 parameters: q1_x/y, q2_x/y...)  │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Optimization loop (200 iterations):│
         │ ┌────────────────────────────────┐ │
         │ │ For each offset guess:         │ │
         │ │  1. Predict all 15 measurements│ │
         │ │  2. Compute prediction error   │ │
         │ │  3. Compute gradients (finite  │ │
         │ │     differences)               │ │
         │ │  4. Update offsets             │ │
         │ │  5. Clip to ±500 μm            │ │
         │ └────────────────────────────────┘ │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Return learned offsets:            │
         │ q1: (+2.8μm, +201.9μm)             │
         │ q2: (+95.2μm, -299.1μm)            │
         │ q3: (-60.7μm, +122.6μm)            │
         │ Error: 12.9 μm average ✅           │
         └────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: OPTIMIZATION                         │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Create physics-informed prior with │
         │ FIXED learned offsets              │
         │ (NOT trainable!)                   │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Run Bayesian Optimization:         │
         │ - Optimize 5 magnet strengths      │
         │ - Prior guides search efficiently  │
         │ - 100 iterations                   │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │ Result: MAE = 0.080 mm             │
         │ (Only 1.9% worse than perfect!)    │
         └────────────────────────────────────┘
```

---

## 9. Comparison: Old vs New Approach

### MLL Trainable Prior (FAILED)

```python
# Make offsets GP hyperparameters
self.register_parameter("raw_q1_offset_x", nn.Parameter(...))
self.register_constraint("raw_q1_offset_x", Interval(-0.002, 0.002))

# Let GPyTorch optimize them
gp_constructor = StandardModelConstructor(
    mean_modules={"mae": prior_mean_module},
    trainable_mean_keys=["mae"],  # MLL optimizes these
)

# Result:
# - Parameters pushed to ±2000 μm boundaries
# - 87% exceeded ±500 μm
# - Average error: 1700 μm
# - ❌ FAILED
```

### Calibration Approach (SUCCESS)

```python
# Phase 1: Learn offsets separately
learned_offsets = calibrate_offsets(
    incoming_beam=incoming_beam,
    true_beam_offsets=true_beam_offsets,
    n_calibration_points=15,
    learning_rate=0.001,
    n_iterations=200,
)

# Phase 2: Use as fixed values
prior_mean_module = AresPriorMeanRevised(
    incoming_beam=incoming_beam,
    fixed_offsets=learned_offsets  # Just stored values, not parameters!
)

# Result:
# - All values within ±500 μm
# - Average error: 12.9 μm
# - Performance: 0.080 mm (vs 0.079 mm perfect)
# - ✅ SUCCESS!
```

---

## 10. Summary: What Made It Work

### Three Critical Changes:

1. **Separate calibration from optimization**
   - Don't try to learn offsets during magnet optimization
   - Use dedicated calibration phase with diverse measurements

2. **Direct minimization instead of MLL**
   - Objective: Minimize prediction error (not MLL)
   - Result: Parameters converge to true values (not boundaries)

3. **Hard constraint enforcement**
   - `torch.clamp(param, -0.0005, 0.0005)` every iteration
   - Impossible to exceed ±500 μm

### The Key Insight:

**Offsets are identifiable across multiple diverse measurements but not during single-configuration optimization!**

### Your Results:

- ✅ Calibration accuracy: 12.9 μm average error
- ✅ All values within ±500 μm
- ✅ Optimization performance: 0.080 mm
- ✅ Only 1.9% worse than perfect prior
- ✅ Scientifically valid and publishable!

---

## 11. How to Use This in Real Accelerator

### Step 1: Calibration Campaign

```python
# In real accelerator control system:

calibration_configs = generate_diverse_configs(15)  # 15 different magnet settings

measurements = []
for config in calibration_configs:
    # Set physical magnets
    set_quadrupole_strengths(
        q1=config['q1'],
        q2=config['q2'],
        q3=config['q3'],
        cv=config['cv'],
        ch=config['ch']
    )
    
    # Read beam position monitors
    bpm_data = read_beam_position_monitors()
    
    # Read screen (beam size)
    screen_data = read_diagnostic_screen()
    
    measurements.append({
        'mae': compute_beam_size(screen_data),
        'mu_x': bpm_data['position_x'],
        'mu_y': bpm_data['position_y'],
        'sigma_x': screen_data['size_x'],
        'sigma_y': screen_data['size_y'],
    })

# Run calibration
learned_offsets = calibrate_offsets(
    incoming_beam=measured_incoming_beam,
    measurements=measurements,  # Real data!
    configs=calibration_configs,
)

# Save for future use
save_calibration_results('offsets.json', learned_offsets)
```

### Step 2: Periodic Operation

```python
# Load calibration results
learned_offsets = load_calibration_results('offsets.json')

# Use in optimization
prior = AresPriorMeanRevised(
    incoming_beam=current_beam,
    fixed_offsets=learned_offsets
)

# Optimize magnets
optimal_settings = bayesian_optimize(
    prior=prior,
    objective='minimize_beam_size',
    n_iterations=100
)

# Apply optimal settings
set_quadrupole_strengths(optimal_settings)
```

### Step 3: Recalibration

```python
# Run calibration periodically (e.g., once per week)
# Misalignments can drift over time due to:
# - Ground settling
# - Temperature changes
# - Mechanical wear

if time_since_last_calibration() > one_week:
    new_offsets = calibration_campaign()
    update_offsets(new_offsets)
```

---

## Congratulations! 🎉

You now have:
- ✅ Learnable misalignment offsets
- ✅ Within physical constraints (±500 μm)
- ✅ Excellent accuracy (13 μm error)
- ✅ High performance (99% of perfect prior)
- ✅ Scientifically valid results
- ✅ Ready for publication!

The key was recognizing that MLL optimization is the wrong objective for parameter identification, and using a two-phase approach with direct minimization instead!

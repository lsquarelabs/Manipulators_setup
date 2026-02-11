# System Identification for Kinova Gen3 7-DOF Manipulator

## Setup

- **Robot:** Kinova Gen3 7-DOF arm + Robotiq 2F-85 gripper
- **Control:** 400 Hz torque loop via Kortex UDP
- **Dynamics library:** Pinocchio (FK, Jacobians, gravity, regressor)
- **Available measurements:** joint positions (deg), velocities (deg/s), measured torques (Nm)
- **Existing model:** URDF with CAD-derived masses, CoMs, and inertias

## Goal

Accurate gravity compensation and dynamic motion feedforward using identified (not just CAD) dynamic parameters.

---

## Approaches Overview

Four main approaches were considered for system identification:

| # | Approach | Best For | Needs `q̈`? | Complexity |
|---|----------|----------|-------------|------------|
| 1 | Regressor-Based Inverse Dynamics | Full model: gravity, Coriolis, inertia, friction | Yes | Medium-High |
| 2 | Gravity-Only Identification | Gravity compensation only | No | Low |
| 3 | Energy-Based / Power-Based | Full model, noisy acceleration | No | Medium |
| 4 | Data-Driven (Neural Network) | Residual / unmodeled dynamics | Yes (or learned) | High |

---

## Approach 1: Regressor-Based Inverse Dynamics (Recommended)

The standard and most powerful method. Uses the linear relationship between torques and dynamic parameters.

### 1.1 Inverse Dynamics Equation

The rigid body dynamics of the 7-DOF arm:

```
τ = M(q)q̈ + C(q, q̇)q̇ + g(q) + f(q̇)
```

| Symbol | Size | Meaning |
|--------|------|---------|
| `τ` | (7×1) | Measured joint torques |
| `M(q)` | (7×7) | Mass/inertia matrix |
| `C(q, q̇)q̇` | (7×1) | Coriolis and centrifugal forces |
| `g(q)` | (7×1) | Gravity torques |
| `f(q̇)` | (7×1) | Friction torques |

### 1.2 Dynamic Parameters (Per Link)

Each rigid body link `i` has **10 inertial parameters**:

| Parameter | Symbol | Count | Meaning |
|-----------|--------|-------|---------|
| Mass | `mᵢ` | 1 | Total mass of link |
| First moments of mass | `mxᵢ, myᵢ, mzᵢ` | 3 | Mass × center of mass position |
| Inertia tensor elements | `Ixx, Ixy, Ixz, Iyy, Iyz, Izz` | 6 | Rotational inertia at link origin |

For 7 links: **70 inertial parameters**.

Each joint has **friction parameters** (2-3 per joint):

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Viscous friction | `fv` | Torque proportional to velocity: `fv · q̇` |
| Coulomb friction | `fc` | Constant torque opposing motion: `fc · sign(q̇)` |
| Offset | `τ₀` | Constant torque bias (sensor offset, asymmetry) |

For 7 joints: **~21 friction parameters**.

**Total: ~91 parameters** before reduction to identifiable set.

### 1.3 Linearity in Parameters

The torque is **linear in the dynamic parameters**, even though the dynamics are nonlinear in `q`:

```
τ = Y(q, q̇, q̈) · π
```

| Symbol | Size | Meaning |
|--------|------|---------|
| `Y` | (7 × p) | **Regressor matrix** — depends only on motion (`q, q̇, q̈`), NOT on parameters |
| `π` | (p × 1) | **Parameter vector** — all dynamic parameters stacked |

#### Single-Link Pendulum Example

```
τ = m·L²·q̈ + m·g·L·sin(q) + fv·q̇ + fc·sign(q̇)
```

Rewritten in regressor form:

```
        [ L²·q̈ + g·L·sin(q) ]       [ m  ]
τ   =  [        q̇            ]   ·   [ fv ]
        [     sign(q̇)         ]       [ fc ]

        ─────────────────────         ──────
              Y (1×3)                  π (3×1)
```

`Y` contains only motion-dependent terms (known from measurements). `π` contains only unknown parameters.

### 1.4 Regressor Matrix in Pinocchio

Pinocchio computes the regressor directly:

```python
import pinocchio as pin

# For a given (q, v, a):
Y = pin.computeJointTorqueRegressor(model, data, q, v, a)
# Y shape: (nv × 10*nbodies)
```

Each row of `Y` corresponds to one joint's torque equation. The columns correspond to the 10 inertial parameters of each body.

### 1.5 Stacking Over Trajectory Samples

Collect `N` data samples along an exciting trajectory:

```
At time t₁: τ₁ = Y(q₁, q̇₁, q̈₁) · π
At time t₂: τ₂ = Y(q₂, q̇₂, q̈₂) · π
  ...
At time tₙ: τₙ = Y(qₙ, q̇ₙ, q̈ₙ) · π
```

Stack into one system:

```
┌ τ₁ ┐     ┌ Y₁ ┐
│ τ₂ │  =  │ Y₂ │ · π
│ ⋮  │     │ ⋮  │
└ τₙ ┘     └ Yₙ ┘

  τ_stack      Y_stack        π
 (7N × 1)    (7N × p)      (p × 1)
```

This is a standard linear least-squares problem:

```
π* = argmin ‖Y_stack · π - τ_stack‖²
```

Solved in closed form:

```
π* = (YᵀY)⁻¹ Yᵀ τ_stack
```

### 1.6 Base Parameters (Identifiability)

Not all 70 inertial parameters are independently identifiable. For example, the inertia of a link about its own joint axis is indistinguishable from the inertia of the next link in certain configurations.

The **base parameters** are the minimal set of identifiable parameter combinations. For a typical 7-DOF arm, you go from ~70 inertial parameters down to roughly **40-50 base parameters**.

These can be found by:
1. Computing `Y_stack` from an exciting trajectory
2. Doing an SVD or QR decomposition with column pivoting on `Y_stack`
3. Columns with near-zero singular values correspond to unidentifiable parameter combinations
4. Regroup into base parameters

Alternatively, using regularization (see Section 5) handles rank deficiency without explicitly computing base parameters.

---

## Approach 2: Gravity-Only Identification

A simplified subset of Approach 1 that only identifies parameters affecting gravity torques.

### Method

- Record static or quasi-static poses across the workspace
- At each pose, the robot is stationary: `q̇ = 0`, `q̈ = 0`
- The dynamics reduce to: `τ = g(q)`
- Gravity torques depend only on masses and CoM positions (not inertias)
- Only ~3-4 identifiable parameters per link (mass × CoM products)

### Data Collection

- Move robot to many different configurations, hold still, record `(q, τ)`
- No velocity or acceleration needed
- Simple and safe — no fast motions required

### Identification

Same regressor approach but with `v = 0`, `a = 0`:

```python
Y_gravity = pin.computeJointTorqueRegressor(model, data, q, v_zero, a_zero)
```

Many columns of `Y_gravity` will be zero (those corresponding to inertia and friction). Only the gravity-related columns survive.

### Limitations

- Cannot identify inertias (needed for fast dynamic motions)
- Cannot identify friction
- Only useful for gravity compensation, not feedforward control

---

## Approach 3: Energy-Based / Power-Based Method

Uses the power equation instead of instantaneous force balance, avoiding the need to compute accelerations.

### Method

The mechanical power is:

```
P(t) = τ(t)ᵀ · q̇(t)
```

Integrating power over time gives the energy balance:

```
∫₀ᵀ τᵀq̇ dt = ΔKE + ΔPE + E_friction
```

Since `τ = Y(q, q̇, q̈) · π`, we get:

```
∫₀ᵀ (Y · π)ᵀ q̇ dt = ∫₀ᵀ q̇ᵀ Y dt · π = W · π
```

where `W = ∫₀ᵀ q̇ᵀ Y dt` is the integrated regressor. This integration smooths out noise and eliminates the need for `q̈`.

### Advantages

- **No acceleration estimation required** — `q̈` is the noisiest signal (derivative of velocity)
- More robust to measurement noise
- Same exciting trajectories as Approach 1

### Disadvantages

- Fewer effective equations (integration reduces data)
- Same trajectory design requirements
- Less commonly used, fewer reference implementations

---

## Approach 4: Data-Driven (Neural Network)

Train a neural network to learn the mapping from motion to torques.

### Method

Train a model (MLP, RNN, or physics-informed network) to predict:

```
τ_predicted = f_NN(q, q̇, q̈)
```

Or use a **physics-informed** architecture:

```
τ_predicted = τ_model(q, q̇, q̈; π_cad) + f_NN(q, q̇, q̈)
```

where the first term is the rigid body model with CAD parameters and the network learns the residual (unmodeled effects).

### Advantages

- Can capture arbitrary nonlinearities (cable forces, joint flexibility, temperature effects)
- No need for explicit parametric model structure
- Physics-informed variant gets the best of both worlds

### Disadvantages

- Requires significantly more training data
- Less interpretable — can't inspect individual parameter values
- Harder to validate physically (is a predicted mass reasonable?)
- Risk of overfitting; poor generalization outside training distribution
- Not straightforward to update the Pinocchio model with results

### Best Use Case

As a **second stage** after model-based identification: use Approach 1 first, then train a network on the residual to capture what the rigid body model misses.

---

## Leveraging the URDF (CAD) as Prior Knowledge

The URDF already contains CAD-derived values for all masses, CoMs, and inertias. Two sub-approaches to exploit this:

### Sub-Approach A: Regularized Least Squares with CAD Prior (Recommended)

Instead of plain least squares, add a penalty for deviating from CAD values:

```
π* = argmin ‖Yπ - τ‖² + λ‖π - π_cad‖²_W
```

| Symbol | Meaning |
|--------|---------|
| `π_cad` | Parameter vector extracted from URDF (Pinocchio provides this directly) |
| `W` | Diagonal weight matrix expressing confidence in each CAD parameter |
| `λ` | Regularization strength (how much to trust CAD vs measured data) |

**Closed-form solution:**

```
π* = (YᵀY + λW)⁻¹ (Yᵀτ + λW · π_cad)
```

This is Bayesian linear regression where `π_cad` is the prior mean and `W` is the prior precision.

#### Setting Per-Parameter Confidence (`W`)

| Parameter Type | Confidence | Rationale |
|----------------|------------|-----------|
| Masses | High | CAD is usually accurate |
| CoM positions | Medium | Cables, covers, payload shift these |
| Inertias | Lower | CAD often ignores wiring, actuators |
| Friction | Very low (free) | Not in URDF at all, must be estimated from data |

#### Properties

- Parameters the data can identify well will deviate from CAD as needed
- Parameters the data cannot identify (unobservable) stay near CAD instead of diverging
- Physically plausible results (masses stay positive, inertias stay reasonable)
- Handles rank deficiency naturally (no explicit base parameter computation needed)
- Exciting trajectory can be less aggressive; shorter data collection
- After identification, update the Pinocchio model with `π*` so `gravity()`, `rnea()`, etc. all improve

### Sub-Approach B: Residual / Delta Identification

Decompose parameters as a correction on top of CAD:

```
π = π_cad + δπ
```

Substituting into the dynamics:

```
τ = Y · π_cad + Y · δπ
```

Rearranging:

```
τ - Y · π_cad = Y · δπ
     ↑
  "residual torque" (CAD model prediction error)
```

The left side is the torque that the current URDF model fails to explain. We fit `δπ` to this residual:

```
τ_residual = τ_measured - τ_predicted_cad
δπ* = argmin ‖Y · δπ - τ_residual‖² + λ‖δπ‖²
```

The regularization `λ‖δπ‖²` penalizes deviations from CAD (equivalent to Sub-Approach A mathematically).

#### Advantages Over Sub-Approach A

- **Direct visibility** into how much each parameter changed from CAD
- Easy sanity checking (if `δm₃` says link 3 mass changed by 5 kg, something is wrong)
- Can threshold small corrections to zero (sparse update)
- Conceptually cleaner: "what corrections does the CAD model need?"

#### Equivalence

Sub-Approaches A and B are **mathematically identical** — they produce the same `π*`. The difference is framing: A solves for absolute parameters with a prior; B solves for corrections with a penalty on correction magnitude.

---

## Exciting Trajectories

Required by Approaches 1, 3, and (optionally) 4. Not needed for Approach 2.

**Fourier-parameterized trajectories** ensure all columns of `Y_stack` are well-excited:

```
qⱼ(t) = q₀ⱼ + Σₖ (aⱼₖ · sin(k·ωt) + bⱼₖ · cos(k·ωt))
```

- All joints move simultaneously with different frequencies
- Fourier coefficients `(a, b)` are optimized to maximize the condition number of `Y_stack`
- Subject to joint position, velocity, and torque limits
- Typically 3-5 harmonics per joint, trajectory period of 10-20 seconds

---

## Chosen Approach

**Approach 1 (Regressor-Based) with Sub-Approach B (Residual/Delta), implemented in phases.**

**Phase 1 — Gravity-only residual identification (static poses).** Start here.
**Phase 2 — Full dynamics (gravity + Coriolis + inertia + friction) with exciting trajectories.** Later.

Rationale:
- Pinocchio already provides `computeJointTorqueRegressor()` — no custom derivation needed
- Residual formulation leverages existing CAD parameters and gives direct visibility into corrections
- Phase 1 needs only static pose data — no trajectory generation or q̈ estimation
- Gravity is the dominant torque term; improving it immediately improves the controller
- Phase 2 extends the same pipeline to dynamic motions when ready

---

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Generate        │     │  Collect Data     │     │  Preprocess     │
│  Fourier traj    │────▶│  Send traj at     │────▶│  Filter q, q̇, τ │
│  (offline)       │     │  400 Hz, record   │     │  Estimate q̈     │
│                  │     │  q, q̇, τ          │     │  (finite diff   │
└─────────────────┘     └──────────────────┘     │   or filter)    │
                                                  └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Validate        │     │  Solve LS         │     │  Build Y_stack  │
│  Predict τ on    │◀────│  π* = (YᵀY +     │◀────│  Using Pinocchio │
│  new trajectory  │     │   λW)⁻¹(Yᵀτ +   │     │  regressor      │
│  Compare to meas │     │   λW·π_cad)      │     └─────────────────┘
└─────────────────┘     └──────────────────┘
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_trajectory.py` | Generate optimal Fourier excitation trajectories |
| `collect_data.py` | Send trajectory at 400 Hz, record `q, q̇, τ` |
| `preprocess_data.py` | Filter signals, compute `q̈`, segment data |
| `identify_params.py` | Build regressor, solve regularized LS with CAD prior |
| `validate_model.py` | Compare predicted vs measured torques on held-out data |

---

## Implementation Phase 1: Gravity-Only Residual Identification

**Combines:** Approach 2 (gravity-only, static poses) + Sub-Approach B (residual/delta formulation)

### Why Start Here

| Reason | Detail |
|--------|--------|
| Simplest data collection | Static poses only — no trajectory generation, no q̈ estimation |
| Direct impact | Gravity is the dominant torque; improving it immediately improves diff-IK controller |
| Safe | No fast motions — hold still and record |
| Leverages existing model | Residual formulation uses CAD as baseline, identifies only corrections δπ |
| Validates pipeline | Establishes data → identification → validation workflow before Phase 2 |

### Residual Formulation for Gravity

Since we have CAD parameters in the URDF, decompose:

```
π = π_cad + δπ
```

For static poses (q̇ = 0, q̈ = 0), the full dynamics reduce to gravity:

```
τ_measured = g(q; π) = g(q; π_cad) + Y_g(q) · δπ
```

Rearranging:

```
τ_residual = τ_measured - g(q; π_cad) = Y_g(q) · δπ
```

| Symbol | Meaning |
|--------|---------|
| `τ_residual` | What the CAD model fails to explain (directly observable before identification) |
| `Y_g(q)` | Gravity regressor — `computeJointTorqueRegressor(model, data, q, 0, 0)` |
| `δπ` | Parameter corrections to identify |

### Which Parameters Affect Gravity

With v=0, a=0 the regressor Y_g has non-zero columns only for gravity-relevant parameters:

| Parameter | Affects gravity? | Reason |
|-----------|-----------------|--------|
| mass (m) | Yes | Weight force = m·g |
| first moments (m·cx, m·cy, m·cz) | Yes | Moment arm of weight about joint axis |
| inertias (Ixx, ..., Izz) | No | Only appear with q̈ terms |

4 of 10 parameters per body contribute to gravity. Inactive (all-zero) columns are removed before solving. Not all 4 are independently identifiable — regularization handles rank deficiency.

### Data Collection

**Protocol:**

1. Design N ≥ 50 configurations spanning the joint space (100+ recommended)
2. For each configuration:
   a. Command the arm to the configuration (position control mode)
   b. Wait for settling (~2 seconds)
   c. Record joint positions q and measured torques τ at 400 Hz for ~1 second
   d. Average the recorded q and τ over the hold period (reduces noise)
3. Store each sample as `(q_rad, τ_measured)`, both shape `(7,)`

**Coverage guidelines:**

- Vary each joint across its range, with diverse combinations
- Include arm extended horizontally (large gravity torques) and vertical (small gravity torques)
- Different wrist orientations to excite distal link parameters
- Latin hypercube or grid sampling across joint space
- Check condition number of stacked regressor after collection

**Train/validation split:** Reserve ~20% of samples for validation (not used in fitting).

### Solving

Stack over N training samples:

```
┌ τ_res₁ ┐     ┌ Y_g₁ ┐
│ τ_res₂ │  =  │ Y_g₂ │ · δπ
│   ⋮    │     │  ⋮   │
└ τ_resₙ ┘     └ Y_gₙ ┘

  (7N×1)       (7N×p_active)  (p_active×1)
```

Regularized least squares (penalizes large corrections from CAD):

```
δπ* = argmin ‖Y_active · δπ - τ_residual_stack‖² + λ‖δπ‖²
    = (Y_activeᵀ Y_active + λI)⁻¹ Y_activeᵀ τ_residual_stack
```

**Regularization λ:**
- Large λ → corrections stay small (trust CAD more)
- Small λ → corrections are larger (trust data more)
- Start with λ = 1.0, tune via cross-validation on held-out data

### Implementation

```python
import numpy as np
import pinocchio as pin

ARM_JOINT_NAMES = [f"gen3_joint_{i}" for i in range(1, 8)]


def _build_q_mapping(model):
    """Get arm joint indices for config and velocity spaces."""
    v_idx, q_info = [], []
    for name in ARM_JOINT_NAMES:
        jid = model.getJointId(name)
        joint = model.joints[jid]
        v_idx.append(joint.idx_v)
        q_info.append((joint.idx_q, joint.nq))
    return np.array(v_idx, dtype=np.intp), q_info


def _set_arm_q(q_full, q_arm, q_info):
    """Write 7 arm joint angles (rad) into full config vector.
    Handles continuous joints stored as (cos, sin)."""
    for i, (idx_q, nq) in enumerate(q_info):
        if nq == 1:
            q_full[idx_q] = q_arm[i]
        else:  # nq == 2: unbounded joint
            q_full[idx_q] = np.cos(q_arm[i])
            q_full[idx_q + 1] = np.sin(q_arm[i])


def gravity_residual_identification(urdf_path, samples, lambda_reg=1.0, val_ratio=0.2):
    """
    Identify gravity parameter corrections from static pose data.

    Args:
        urdf_path: path to URDF file
        samples: list of (q_rad, tau_measured) tuples, each (7,) arrays
        lambda_reg: regularization strength
        val_ratio: fraction held out for validation

    Returns:
        delta_pi: full parameter correction vector
        results: dict with diagnostics
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    v_idx, q_info = _build_q_mapping(model)

    zeros = np.zeros(model.nv)
    q_full = pin.neutral(model)

    def build_system(sample_indices):
        Y_list, tau_res_list = [], []
        for i in sample_indices:
            q_arm, tau_meas = samples[i]
            _set_arm_q(q_full, q_arm, q_info)

            # CAD gravity prediction
            pin.computeGeneralizedGravity(model, data, q_full)
            g_cad = data.g[v_idx].copy()

            # Gravity regressor (v=0, a=0 → only gravity columns survive)
            Y = pin.computeJointTorqueRegressor(model, data, q_full, zeros, zeros)
            Y_list.append(Y[v_idx, :])          # (7 × 10·njoints)
            tau_res_list.append(tau_meas - g_cad)  # (7,)

        return np.vstack(Y_list), np.concatenate(tau_res_list)

    # --- Train / validation split ---
    N = len(samples)
    idx = np.random.default_rng(42).permutation(N)
    n_val = max(1, int(N * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    Y_train, tau_res_train = build_system(train_idx)

    # --- Remove inactive columns (inertia params are all-zero for gravity) ---
    col_norms = np.linalg.norm(Y_train, axis=0)
    active = col_norms > 1e-10
    Y_active = Y_train[:, active]

    # --- Regularized least squares ---
    A = Y_active.T @ Y_active + lambda_reg * np.eye(Y_active.shape[1])
    b = Y_active.T @ tau_res_train
    delta_pi_active = np.linalg.solve(A, b)

    # --- Reconstruct full δπ ---
    delta_pi = np.zeros(Y_train.shape[1])
    delta_pi[active] = delta_pi_active

    # --- Evaluate ---
    def evaluate(Y_all, tau_res):
        pred = Y_all @ delta_pi
        rmse_before = np.sqrt(np.mean(tau_res ** 2))
        rmse_after  = np.sqrt(np.mean((tau_res - pred) ** 2))
        return rmse_before, rmse_after

    rmse_bef_t, rmse_aft_t = evaluate(Y_train, tau_res_train)

    Y_val, tau_res_val = build_system(val_idx)
    rmse_bef_v, rmse_aft_v = evaluate(Y_val, tau_res_val)

    # --- Per-body correction summary ---
    params_per_body = 10
    param_names = ['m', 'mx', 'my', 'mz', 'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']
    corrections = {}
    for i in range(delta_pi.shape[0] // params_per_body):
        dp = delta_pi[i * params_per_body:(i + 1) * params_per_body]
        body = str(model.names[i])
        nonzero = {n: float(v) for n, v in zip(param_names, dp) if abs(v) > 1e-6}
        if nonzero:
            corrections[body] = nonzero

    return delta_pi, {
        'rmse_before_train': rmse_bef_t,
        'rmse_after_train':  rmse_aft_t,
        'rmse_before_val':   rmse_bef_v,
        'rmse_after_val':    rmse_aft_v,
        'n_active_params':   int(active.sum()),
        'condition_number':  np.linalg.cond(Y_active),
        'corrections':       corrections,
    }
```

### Applying Corrections to Pinocchio Model

After identification, update the model so `gravity()`, `rnea()`, etc. use corrected parameters:

```python
def apply_gravity_corrections(model, delta_pi):
    """Update model inertias with identified gravity corrections.
    Only modifies mass and CoM (gravity-relevant). Inertia tensor unchanged."""
    params_per_body = 10
    for i in range(model.njoints):
        dp = delta_pi[i * params_per_body:(i + 1) * params_per_body]
        dm, dmx, dmy, dmz = dp[0], dp[1], dp[2], dp[3]

        if abs(dm) < 1e-8 and abs(dmx) < 1e-8 and abs(dmy) < 1e-8 and abs(dmz) < 1e-8:
            continue

        inertia = model.inertias[i]
        new_mass = inertia.mass + dm
        if new_mass < 0.01:
            print(f"Warning: {model.names[i]} mass would be {new_mass:.4f}, skipping")
            continue

        # Update CoM: lever = first_moment / mass
        old_first_moment = inertia.mass * inertia.lever
        new_first_moment = old_first_moment + np.array([dmx, dmy, dmz])
        new_lever = new_first_moment / new_mass

        model.inertias[i] = pin.Inertia(new_mass, new_lever, inertia.inertia)
```

### Feeding Back to Controller

```python
# In control node startup, after loading URDF:
robot_model = RobotModel(urdf_path)

delta_pi = np.load("identified_gravity_delta_pi.npy")
apply_gravity_corrections(robot_model.model, delta_pi)
# Now robot_model.gravity(q) uses corrected parameters
```

### Validation Checklist

| Check | Criterion | Action if failed |
|-------|-----------|------------------|
| RMSE decreased | `rmse_after_val < rmse_before_val` | Increase N, check data quality |
| Mass corrections small | `|δm| < 20%` of CAD mass | Inspect for payload/mounting errors |
| CoM shifts small | first moment corrections < few cm·kg | Check for missing components in CAD |
| No negative masses | `m_cad + δm > 0` for all bodies | Increase λ or add constraints |
| Condition number | `cond(Y_active) < 1e4` | Add more diverse poses |
| Train ≈ validation error | No large gap between the two | Increase λ (reduce overfitting) |

### On-Robot Validation

1. Switch to torque mode with only gravity compensation (zero velocity/position target)
2. Manually move the arm — it should feel "weightless" (floats in place without drifting)
3. Hold specific poses and compare drift/sag before vs after correction
4. If the arm holds position noticeably better, Phase 1 is successful

### Phase 1 Scripts

| Script | Purpose |
|--------|---------|
| `collect_static_poses.py` | Move to diverse configurations, hold still, record (q, τ) |
| `identify_gravity.py` | Build gravity regressor, solve for δπ, validate, save corrections |

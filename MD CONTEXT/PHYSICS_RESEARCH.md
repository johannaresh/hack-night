# Physics Engine Research & Design Plan

## Context

Hackathon task: build a lightweight but realistic quadrotor physics engine (`physics.py`, `config.py`) to support RL training for autonomous FPV drone flight. The engine must:
- Stay minimal (6-hour constraint, NumPy-only)
- Remain **fast** (state is contiguous flat array, vectorizes well)
- Be **trainable** (PPO agent learns to fly toward camera bboxes)
- Honor the README interface contract (action/observation signatures are team commitments)

## Research Findings

### Open-Source Reference Survey
Reviewed 6 well-known drone physics simulators to understand the fidelity/complexity spectrum:

| Repo | Fidelity | Key Extras | Complexity | License |
|------|----------|-----------|-----------|---------|
| **gym-pybullet-drones** (UTIAsDSL, 2k★) | Full rigid-body + gyroscopic coupling | Per-motor RPM→thrust(KF)/torque(KM), ground effect, quadratic drag, downwash | 1150 lines (BaseAviary) | MIT |
| **PyFlyt** (jjshoots, 252★) | Per-motor thrust allocation + first-order lag | Motor map matrix, quadratic xyz/pqr drag, wind module | 531+546 lines (core modules) | MIT |
| **safe-control-gym** (UTIAsDSL, 900★) | Full symbolic nonlinear EOM (CasADi) | Randomizable inertia, canonical 3D rigid-body equations | 941 lines (quadrotor.py) | MIT |
| **RotorS** (ETH ASL, 1.5k★) | Gazebo + rotor aerodynamics | Rotor drag coeff, rolling moment (gyro), per-motor torque constants | 488+173 lines (C++, Gazebo) | Apache-2.0 |
| **Flightmare** (UZH RPG, 1.4k★) | Full torque/inertia rigid body, cleanest ref | Rotor-speed-based thrust + saturation; `J_inv_ * (T - ω×(J·ω))` | 222 lines (C++) | MIT |
| **AirSim** (Microsoft, 18k★) | Unreal engine 6-DOF | Per-vertex quadratic drag (`drag∝v²`), wind field | 465 lines (FastPhysicsEngine.hpp) | MIT |

**Recommendation from survey:** Two highest-value additions for 30–60 min budget are:
1. **Replace P-controller with real torque/inertia rotational dynamics** (Flightmare/gym-pybullet-drones style): `rpy_rates_deriv = J_inv @ (τ - ω × (J @ ω))`. Unlocks gyroscopic coupling, realistic angular acceleration limits, eliminates instant rate tracking artifact.
2. **Per-motor thrust allocation** (PyFlyt/gym-pybullet-drones): 4 independent motor thrusts → 3-axis torques via frame geometry. Prerequisite for #1; gives realistic yaw authority under saturation.

### Academic Literature (20 papers, Consensus MCP)

**Core physics modeling** — validates our approach:
- **Eschmann 2023** "Learning to Fly in Seconds": lean simulator + domain randomization is sufficient for RL; real strength is batched contiguous arrays (our flat state choice).
- **Kaufmann 2022** "Benchmark Comparison of Learned Control Policies": body-rate + collective thrust commands transfer sim-to-real better than direct rotor RPM.
- **Nekoo 2022** "Quaternion-based Spacecraft Dynamics": quaternion attitude avoids gimbal lock; cost is O(1) renormalization per step.

**Aerodynamic detail** (for domain randomization knobs):
- [1] **NeuroBEM 2021**: Aerodynamic effects negligible at low speeds, dominant at high speeds (65 km/h+). Blade-element momentum theory captures thrust/torque/drag; hybrid neural residual for aggressive acro.
- [2] **Modeling Quadrotor Dynamics in Wind 2021**: Blade element momentum theory + wind coupling; rotor drag exists even without external wind (from drone's own translational motion).
- [8] **Rotor Aerodynamics 2016**: Momentum theory + blade element theory: thrust model, horizontal forces, torque. Geometric blade variations affect thrust-to-force ratio.
- [10] **Vertical Wind Disturbance Response 2026**: Blade Element Momentum Theory Informed (BEMTI) models significantly improve wind-gust response vs. classical simplified thrust equations.
- [16] **Differential Flatness of Quadrotor 2017**: Linear rotor drag effects preserve differential flatness → feedforward control works. Drag coefficients can be identified via gradient-free optimization.

**Motor dynamics & control**:
- [3] **Time-Varying Rotor Aerodynamics 2023**: Blade flapping + dynamic stall effects during forward flight; BET underpredicts thrust without them. Reduced-order model feasible.
- [12] **Nonlinear Dynamic Modeling 2012**: Implicit thrust model using **induced momentum** (inflow effects) more accurate than constant thrust coefficient during dynamic maneuvers.
- [15] **Aerodynamic-Parameter ID & Adaptive LADRC 2021**: First identify aerodynamic params via frequency-domain tools (CIFER); then adaptive disturbance rejection controller for coupling/nonlinearity.
- [18] **Experimental Characterization of Propulsion 2019**: Blade Element Momentum Theory identification from static wind-tunnel + free-flight data.

**Simplifications used by industry** (validates our leanness):
- [20] **Modeling & Implementation 2020**: Alternative methods to model thrust-torque without reaction torque sensor; validated by flight test.
- [17] **Aerodynamic Modeling via Wind-Tunnel 2024**: Multidimensional lookup tables for aerodynamic data work well in practice.

## Design Decisions

### Tier 1: MVP (already implemented)
✅ **State layout** (flat 14-element array): `[pos(3), vel(3), quat(4, scalar-first), ω(3), motor_thrust(1)]`
- Reason: vectorizes over batch, O(1) access, matches Eschmann 2023.
- Quaternion scalar-first `[w,x,y,z]` avoids gimbal lock (Nekoo 2022).

✅ **Action space**: `[thrust_cmd, roll_rate, pitch_rate, yaw_rate]` (collective + body-rate)
- Reason: transfers sim-to-real better than raw RPM (Kaufmann 2022); matches our fake-CV bbox → roll/pitch/yaw command flow.

✅ **Core dynamics** (Newton-Euler):
- Semi-implicit Euler integration, `dt=1/250` (4ms timestep).
- Thrust into world frame via `quat_rotate(q, [0,0,motor])`.
- Gravity `[0,0,-g·m]` + linear velocity damping `lin_drag·vel`.
- Quaternion renormalize each step (cost: 3 ops per step).

✅ **Motor model**: first-order lag `motor' = (cmd - motor) / tau` (tau ~30ms), standard across all references (Flightmare, PyFlyt, gym-pybullet-drones).

✅ **Rate control**: P-controller `ang_accel = kp_rate * (rate_cmd - ω)` — **to upgrade to real torque/inertia if time**.

✅ **Config knobs** (domain randomization): `ct`, `inertia_scale`, `motor_tau_s`, `kp_rate`, `max_body_rate`, `lin_drag`.

### Tier 2: High-Value Upgrades (if 30–60 min remain)

**2a. Real rigid-body rotational dynamics** (15–20 min)
Replace P-rate-controller with:
```
torque_des = kp_rate * (rate_cmd - omega)
ang_accel = J_inv @ (torque_des - omega × (J @ omega))  # Flightmare pattern
```
- Needs: inertia tensor computed in `config.py` (already done), J_inv cached.
- Benefit: gyroscopic coupling, realistic angular acceleration limits, no "instant rate tracking" artifact.
- Reference: [Flightmare 2020](https://github.com/uzh-rpg/flightmare), lines 100–140 in `quadrotor_dynamics.cpp`.

**2b. Per-motor thrust allocation** (20–30 min)
Replace scalar motor thrust with 4 independent motors:
```
# In config: compute mixing matrix from frame geometry (X/H/Plus config)
# Action: still [thrust_des, roll_rate, pitch_rate, yaw_rate] 
# But internally: thrust_des → 4 motor thrusts via mixer (collective + diff yaw)
motors = mixer @ [thrust_des, tau_roll, tau_pitch, tau_yaw]
motors = clip(motors, 0, max_thrust)  # Per-motor saturation
# State: motors = 4-element array (not scalar)
```
- Needs: mixer matrix (3×4 or 4×4, depends on frame config).
- Benefit: realistic yaw authority under saturation; couples with 2a above.
- Reference: [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py#L550), `_compute_motor_commands()`.

**Decision**: Both 2a and 2b are **coupled**; if implementing only one, do 2a first (rotational dynamics) since it gives immediate physics realism. 2b unlocks realistic yaw, but with 2a alone, the P-controller is now applied to desired torques instead of rates — a win anyway.

### Tier 3: Nice-to-have (skip for hackathon)
- Quadratic air drag (AirSim style): requires airspeed-dependent model, wind API. Linear drag sufficient for MVP.
- Ground effect: Hamel force near z=0. Skip unless training shows unrealistic hover.
- Blade flapping: dynamic stall, Blade Element Theory. Skip; overhead not worth for RL.
- Wind disturbances: add via `env.py` config if needed, not physics core.

## Implementation Order

1. **Verify current MVP works**: run `python -m dronegym.physics` self-check (hover, rate tracking, quaternion norm, no mutation).
2. **If time**: implement **2a (rigid-body rotational dynamics)** — 15 lines, high ROI.
3. **If more time**: implement **2b (per-motor mixing)** — 30 lines, couples with 2a.
4. **Branch sync + integration**: johanna/zsun pull `main`, run integration tests (env + RL), catch any contract breaks.

## Verification Checklist

- [ ] Self-check passes: `python -m dronegym.physics`
- [ ] State layout unchanged (teammates' code still works)
- [ ] `gravity_body()` and `velocity_body()` still populate obs indices 7–11
- [ ] PPO training runs without NaNs on initial gym (even dummy reward)
- [ ] Rollback plan: if 2a/2b breaks training, `git revert` to MVP + iterate

## Paper References (Consensus MCP Search Results)

[1] [NeuroBEM: Hybrid Aerodynamic Quadrotor Model](https://consensus.app/papers/details/54fde1151066559d99c1b78a43646286/?utm_source=claude_desktop) (L. Bauersfeld et al., 2021, 147 citations, ArXiv)

[2] [Modeling Quadrotor Dynamics in a Wind Field](https://consensus.app/papers/details/f7074ad9623155359862a7b43e399ae5/?utm_source=claude_desktop) (Heegyun Jeon et al., 2021, 40 citations, IEEE/ASME Transactions on Mechatronics)

[3] [Time Varying Rotor Aerodynamics for Quadrotor Vehicles](https://consensus.app/papers/details/64b4d6e6cd835548950e35fb8a773120/?utm_source=claude_desktop) (C. Smith et al., 2023, 5 citations, AIAA SCITECH 2023 Forum)

[8] [Aerodynamics of Rotor Blades for Quadrotors](https://consensus.app/papers/details/0043b0ef85a858ff821d2e5996791e80/?utm_source=claude_desktop) (Moses Bangura et al., 2016, 50 citations, arXiv: Fluid Dynamics)

[10] [Vertical Wind Disturbance Response of a Quadrotor With Suspended Payload Using Blade Element Momentum Theory Informed Rotor Models](https://consensus.app/papers/details/b6ef7cb88cd550a2a9b2b7c7ee0a1430/?utm_source=claude_desktop) (R. Jayakumar et al., 2026, 0 citations, IEEE Access)

[12] [Nonlinear Dynamic Modeling for High Performance Control of a Quadrotor](https://consensus.app/papers/details/5de117efb1995122b956782cb251e7fe/?utm_source=claude_desktop) (Moses Bangura et al., 2012, 173 citations)

[15] [Aerodynamic-Parameter Identification and Attitude Control of Quad-Rotor Model with CIFER and Adaptive LADRC](https://consensus.app/papers/details/34276c8bccab5165b05ec56bac99130e/?utm_source=claude_desktop) (Sen Yang et al., 2021, 100 citations, Chinese Journal of Mechanical Engineering)

[16] [Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories](https://consensus.app/papers/details/06a1b8909537548bb62a641dcd0ad605/?utm_source=claude_desktop) (Matthias Faessler et al., 2017, 395 citations, IEEE Robotics and Automation Letters)

[17] [Aerodynamic Modeling and Verification of Quadrotor UAV Using Wind-Tunnel Test](https://consensus.app/papers/details/3796f1b3a92e5d0a8e4f7ae865571167/?utm_source=claude_desktop) (Hoijo Jeong et al., 2024, 3 citations, International Journal of Aeronautical and Space Sciences)

[18] [Experimental Characterization of a Propulsion System for Multi-rotor UAVs](https://consensus.app/papers/details/771fe5750a0b55598cde49ac4e7f0d06/?utm_source=claude_desktop) (Daniele Sartori et al., 2019, 20 citations, Journal of Intelligent & Robotic Systems)

[20] [Modeling and Implementation of Quadcopter Autonomous Flight Based on Alternative Methods to Determine Propeller Parameters](https://consensus.app/papers/details/75e1c655ae24564ca481e98dcc239089/?utm_source=claude_desktop) (Gene Patrick S. Rible et al., 2020, 5 citations, ArXiv)

"""Configuration and derived physical properties for a quadrotor."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DroneConfig:
    mass_g: float
    prop_diameter_in: float
    motor_kv: float
    battery_v: float
    cam_angle_deg: float
    frame_size_mm: float

    # Calibration knobs
    ct: float = 0.1
    inertia_scale: float = 1.0
    motor_tau_s: float = 0.03
    kp_rate: float = 20.0
    max_body_rate: float = 14.0
    lin_drag: float = 0.1
    gravity: float = 9.81

    def __post_init__(self) -> None:
        self.mass_kg = self.mass_g / 1000.0
        self.arm_m = self.frame_size_mm / 2000.0

        diameter_m = self.prop_diameter_in * 0.0254
        max_rpm = self.motor_kv * self.battery_v
        rho = 1.225
        per_rotor_thrust = self.ct * rho * (max_rpm / 60.0) ** 2 * diameter_m ** 4
        self.max_thrust_n = 4.0 * per_rotor_thrust

        ixx_iyy = self.inertia_scale * self.mass_kg * self.arm_m ** 2 / 2.0
        izz = self.inertia_scale * self.mass_kg * self.arm_m ** 2
        self.inertia = np.diag([ixx_iyy, ixx_iyy, izz])
        self.inertia_inv = np.linalg.inv(self.inertia)


def randomize_config(cfg: DroneConfig, rng, pct: float = 0.1) -> DroneConfig:
    """Return a copy of cfg with calibration knobs jittered +/- pct.

    Domain randomization for sim-to-real: training across perturbed dynamics
    yields policies robust to the real drone differing from the nominal model.
    Action mapping (max_body_rate) and gravity stay fixed on purpose.
    """
    def jitter(x: float) -> float:
        return x * (1.0 + rng.uniform(-pct, pct))

    return DroneConfig(
        mass_g=jitter(cfg.mass_g),
        prop_diameter_in=cfg.prop_diameter_in,
        motor_kv=cfg.motor_kv,
        battery_v=cfg.battery_v,
        cam_angle_deg=cfg.cam_angle_deg,
        frame_size_mm=cfg.frame_size_mm,
        ct=jitter(cfg.ct),
        inertia_scale=jitter(cfg.inertia_scale),
        motor_tau_s=jitter(cfg.motor_tau_s),
        kp_rate=jitter(cfg.kp_rate),
        max_body_rate=cfg.max_body_rate,
        lin_drag=jitter(cfg.lin_drag),
        gravity=cfg.gravity,
    )

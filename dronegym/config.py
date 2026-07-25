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

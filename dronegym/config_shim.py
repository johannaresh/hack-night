"""Config shim — stands in for moterodiaz's dronegym/config.py.

Loads the preset YAMLs in configs/ and derives the three quantities env.py
actually touches: `mass_kg`, `max_thrust`, `max_rate`. When the real config.py
lands, env.py keeps working as long as those three names survive; if they get
renamed, adapt here rather than in moterodiaz's file.

TWR and MAX_RATE below are placeholder engineering guesses, NOT derived from
prop diameter / motor KV / battery voltage. Deriving them properly is
moterodiaz's job.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

G = 9.81

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Thrust-to-weight ratio per preset.
TWR = {
    "tiny_whoop": 2.5,
    "freestyle_5inch": 8.0,
    "cinelifter": 4.0,
    "longrange_7inch": 5.0,
    "longrange_10inch": 4.5,
}

# Body-rate command authority per preset [rad/s].
MAX_RATE = {
    "tiny_whoop": 12.0,
    "freestyle_5inch": 10.0,
    "cinelifter": 5.0,
    "longrange_7inch": 8.0,
    "longrange_10inch": 6.0,
}

DEFAULT_TWR = 4.0
DEFAULT_MAX_RATE = 8.0


@dataclass(frozen=True)
class DroneConfig:
    """The interface-contract fields, plus the three derived values env.py uses."""

    name: str
    mass_g: float
    prop_diameter_in: float
    motor_kv: float
    battery_v: float
    cam_angle_deg: float
    frame_size_mm: float

    @property
    def mass_kg(self):
        return self.mass_g / 1000.0

    @property
    def max_thrust(self):
        """Maximum collective thrust [N]."""
        return TWR.get(self.name, DEFAULT_TWR) * self.mass_kg * G

    @property
    def hover_thrust(self):
        """Collective thrust [N] that cancels gravity."""
        return self.mass_kg * G

    @property
    def max_rate(self):
        """Body-rate command authority [rad/s], full-scale action -> this."""
        return MAX_RATE.get(self.name, DEFAULT_MAX_RATE)


_FIELD_NAMES = {f.name for f in fields(DroneConfig)}


def load_config(path):
    """Load a preset. Accepts a path to a YAML file or a bare preset name."""
    p = Path(path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        p = CONFIG_DIR / f"{path}.yaml"
    with open(p) as f:
        raw = yaml.safe_load(f)
    unknown = set(raw) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"{p.name}: unknown config fields {sorted(unknown)}")
    return DroneConfig(**raw)


def all_presets():
    """Every preset in configs/, sorted by name."""
    return [load_config(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))]


if __name__ == "__main__":
    print(f"{'preset':<18} {'mass_kg':>8} {'max_thrust':>11} {'TWR':>5} "
          f"{'max_rate':>9} {'cam_deg':>8}")
    for cfg in all_presets():
        assert cfg.max_thrust > cfg.hover_thrust, f"{cfg.name} cannot hover"
        assert cfg.max_rate > 0
        print(f"{cfg.name:<18} {cfg.mass_kg:>8.3f} {cfg.max_thrust:>10.2f}N "
              f"{cfg.max_thrust / cfg.hover_thrust:>5.1f} "
              f"{cfg.max_rate:>7.1f}/s {cfg.cam_angle_deg:>7.0f}")

    assert len(all_presets()) == 5, "expected 5 presets"
    print("\nall config sanity checks passed")

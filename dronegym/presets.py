"""Load the configs/*.yaml presets into moterodiaz's DroneConfig.

Replaces the old config_shim: DroneConfig now derives max_thrust_n, inertia and
motor/rate time constants from real prop/KV/voltage geometry, so the shim's
placeholder TWR and max_rate tables are gone.

The YAML carries a `name` field that DroneConfig has no slot for (derive_params
drops unknown keys), so it is attached afterwards for logging and run naming.
DroneConfig already sets non-field attributes in __post_init__, so this matches
the existing style — but nothing in the RL layer *requires* `.name`; read it
with getattr if a config might be built by hand.
"""

from pathlib import Path

import yaml

from dronegym.physics import derive_params

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(path):
    """Load a preset. Accepts a path to a YAML file or a bare preset name."""
    p = Path(path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        p = CONFIG_DIR / f"{path}.yaml"
    with open(p) as f:
        raw = yaml.safe_load(f)
    cfg = derive_params(raw)          # drops `name`, keeps the dataclass fields
    cfg.name = raw.get("name", p.stem)
    return cfg


def all_presets():
    """Every preset in configs/, sorted by filename."""
    return [load_config(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))]


if __name__ == "__main__":
    print(f"{'preset':<18} {'mass_kg':>8} {'max_thrust':>11} {'TWR':>6} "
          f"{'max_rate':>9} {'cam':>5}")
    for cfg in all_presets():
        hover = cfg.mass_kg * cfg.gravity
        twr = cfg.max_thrust_n / hover
        print(f"{cfg.name:<18} {cfg.mass_kg:>8.3f} {cfg.max_thrust_n:>10.2f}N "
              f"{twr:>6.1f} {cfg.max_body_rate:>7.1f}/s {cfg.cam_angle_deg:>4.0f}")
        assert cfg.max_thrust_n > hover, f"{cfg.name} cannot hover"

    assert len(all_presets()) == 5, "expected 5 presets"
    print("\nall presets load and can hover")

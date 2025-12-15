from .basis import encoding as basis
from .angle import encoding as angle
from .amplitude import encoding as amplitude
from .feature_map import encoding as feature_map
from .hybrid_angle_zz import encoding as hybrid_angle_zz

REGISTRY = {
    "basis": basis,
    "angle": angle,
    "amplitude": amplitude,
    "feature_map": feature_map,
    "hybrid_angle_zz": hybrid_angle_zz
}

def get_encoding(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown encoding: {name}. Available: {list(REGISTRY.keys())}")

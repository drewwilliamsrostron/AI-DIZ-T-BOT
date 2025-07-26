"""Package-wide constants used across Artibot."""

FEATURE_DIMENSION = 16
# Default number of mini-batches before RL loss kicks in
WARMUP_STEPS = 500

__all__ = ["FEATURE_DIMENSION", "WARMUP_STEPS"]

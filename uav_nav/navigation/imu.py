"""IMU reading dataclass and simple noise model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IMUReading:
    """One IMU sample at rate imu_rate_hz."""
    accel_x_ms2: float = 0.0   # Body-frame forward acceleration (m/s²)
    accel_y_ms2: float = 0.0   # Body-frame right acceleration  (m/s²)
    accel_z_ms2: float = 0.0   # Body-frame down acceleration   (m/s²)
    gyro_x_dps: float = 0.0    # Roll rate  (deg/s)
    gyro_y_dps: float = 0.0    # Pitch rate (deg/s)
    gyro_z_dps: float = 0.0    # Yaw rate   (deg/s, positive = clockwise)
    dt_s: float = 0.01         # Sample period (s)


# Typical MEMS IMU noise values (1-sigma)
ACCEL_NOISE_MS2 = 0.05        # 50 mg
GYRO_NOISE_DPS = 0.1          # 0.1 deg/s
ACCEL_BIAS_WALK_MS2 = 0.001   # bias random walk
GYRO_BIAS_WALK_DPS = 0.005

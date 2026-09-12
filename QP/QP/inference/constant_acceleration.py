"""单物体、整段匀加速模型的加权最小二乘估计，只依赖 NumPy。"""

import numpy as np

from constraints.observation import validate_observation
from operators.classical import require_mass
from world.calibration import PlanarCalibration2D
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, PixelObservation2D, Vec2


def fit_constant_acceleration(
    observations: list[PixelObservation2D],
    calibration: PlanarCalibration2D,
    *,
    mass: float | None = None,
    known_acceleration: Vec2 | None = None,
) -> ObjectTrack2D:
    """最小化像素观测的标准化残差平方和，估计 p0、v0 和可选的 a。

    p(t)=p0+v0*dt+1/2*a*dt²；此参数化严格满足整段匀加速运动学。
    默认从观测估计 a（至少 3 个时刻）；给定 a 时估计 p0、v0（至少 2 个）。
    数据必须同物体、时间有序、帧号唯一。sigma_px 为每轴误差尺度。
    不接收真值或未来观测，不进行接触识别、鲁棒估计或相机参数估计。
    """
    minimum = 3 if known_acceleration is None else 2
    if len(observations) < minimum:
        raise ValueError(f"at least {minimum} observations are required")
    if mass is not None:
        require_mass(mass)
    for observation in observations:
        validate_observation(observation)
        if observation.object_id != observations[0].object_id:
            raise ValueError("all observations must have the same object_id")
    if len({o.frame_id for o in observations}) != len(observations):
        raise ValueError("observation frame_id values must be unique")
    timestamps = np.array([o.timestamp for o in observations])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("observation timestamps must strictly increase")
    dt = timestamps - timestamps[0]
    span = float(dt[-1])
    if not np.isfinite(span):
        raise ValueError("observation time span must be finite")
    u = dt / span
    positions = np.array([calibration.to_world(o.center).as_array() for o in observations])
    sigma = np.array([o.sigma_px for o in observations]) * calibration.meters_per_pixel
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("calibrated observation scales must be finite and positive")

    if known_acceleration is None:
        design = np.column_stack([np.ones(len(u)), u, 0.5 * u**2])
        target = positions
    else:
        if not np.isfinite(known_acceleration.as_array()).all():
            raise ValueError("known_acceleration must be finite")
        design = np.column_stack([np.ones(len(u)), u])
        target = positions - 0.5 * dt[:, None]**2 * known_acceleration.as_array()
    # 公共比例不影响最小二乘解，缩放权重以避免非常小的 sigma 导致溢出。
    weights = sigma.min() / sigma
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None], target * weights[:, None], rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("observations do not determine the model reliably")
    p0 = coefficients[0]
    v0 = coefficients[1] / span
    acceleration = coefficients[2] / span**2 if known_acceleration is None else known_acceleration.as_array()
    states = []
    for observation, offset in zip(observations, dt):
        position = p0 + v0 * offset + 0.5 * acceleration * offset**2
        velocity = v0 + acceleration * offset
        if not np.isfinite([*position, *velocity, *acceleration]).all():
            raise ValueError("estimated state exceeds numeric range")
        states.append(ObjectState2D(
            object_id=observation.object_id, frame_id=observation.frame_id,
            timestamp=observation.timestamp, space=CoordinateSystem.WORLD,
            position=Vec2.from_array(position), velocity=Vec2.from_array(velocity),
            acceleration=Vec2.from_array(acceleration), mass=mass,
        ))
    return ObjectTrack2D(observations[0].object_id, "estimated-point-mass", states)

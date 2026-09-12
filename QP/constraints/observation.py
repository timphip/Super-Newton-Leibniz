"""将状态投影回图像，与同一帧的像素观测比较。"""

from math import isclose, isfinite

import numpy as np

from world.calibration import PlanarCalibration2D
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, PixelObservation2D


def validate_observation(observation: PixelObservation2D) -> None:
    if not isfinite(observation.timestamp):
        raise ValueError("observation timestamp must be finite")
    if not isfinite(observation.sigma_px) or observation.sigma_px <= 0:
        raise ValueError("observation sigma_px must be finite and positive")
    if not np.isfinite(observation.center.as_array()).all():
        raise ValueError("observation center must be finite")


def position_observation_residual(
    state: ObjectState2D,
    observation: PixelObservation2D,
    calibration: PlanarCalibration2D | None = None,
) -> np.ndarray:
    """返回投影位置减观测位置的原始二维像素残差，尚未除以 sigma_px。"""
    validate_observation(observation)
    if state.object_id != observation.object_id or state.frame_id != observation.frame_id:
        raise ValueError("state and observation must match object_id and frame_id")
    if not isfinite(state.timestamp) or not isclose(
        state.timestamp, observation.timestamp, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("state and observation timestamps must match")
    if state.space == CoordinateSystem.WORLD:
        if calibration is None:
            raise ValueError("WORLD observations require explicit calibration")
        predicted = calibration.to_pixel(state.position)
    elif state.space == CoordinateSystem.PIXEL:
        predicted = state.position
    else:
        raise ValueError("unsupported coordinate system")
    if not np.isfinite(predicted.as_array()).all():
        raise ValueError("state position must be finite")
    return predicted.as_array() - observation.center.as_array()


def observation_residual_vector(
    track: ObjectTrack2D,
    observations: list[PixelObservation2D],
    calibration: PlanarCalibration2D | None = None,
) -> np.ndarray:
    """一一对应、同序的 N 个状态/观测返回 2N 个无量纲残差；不跳过缺测。"""
    if not observations or len(track.states) != len(observations):
        raise ValueError("track and observations must have the same nonzero length")
    residuals = []
    for state, observation in zip(track.states, observations):
        if state.object_id != track.object_id:
            raise ValueError("state object_id must match track.object_id")
        residuals.append(position_observation_residual(state, observation, calibration) / observation.sigma_px)
    return np.concatenate(residuals)

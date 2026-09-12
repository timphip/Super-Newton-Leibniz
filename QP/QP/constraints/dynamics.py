"""相邻二维状态的动力学残差。

位置和速度使用区间内加速度保持为 state0.acceleration 的模型。
残差统一采用“后一状态的实际值减去预测值”，返回 shape (2,) 的浮点数组。
这里只计算原始残差，不做加权、平方或求和，也不修改输入状态。
"""

from __future__ import annotations

from math import isfinite

import numpy as np

from operators.kinematics import propagate_position, propagate_velocity
from world.schema import ObjectState2D, Vec2


def _time_delta(state0: ObjectState2D, state1: ObjectState2D) -> float:
    """检查两个状态属于同一物体、同一坐标系，且时间严格递增。"""
    if state0.object_id != state1.object_id:
        raise ValueError("states must have the same object_id")
    if state0.space != state1.space:
        raise ValueError("states must use the same coordinate system")
    if not isfinite(state0.timestamp) or not isfinite(state1.timestamp):
        raise ValueError("timestamps must be finite")

    dt = state1.timestamp - state0.timestamp
    if not isfinite(dt) or dt <= 0:
        raise ValueError("timestamp difference must be finite and greater than 0")
    return dt


def _require_vector(value: Vec2 | None, name: str) -> Vec2:
    """必要的向量必须存在且分量有限；缺失值不代表零。"""
    if value is None:
        raise ValueError(f"{name} is required")
    if not isfinite(value.x) or not isfinite(value.y):
        raise ValueError(f"{name} must contain finite components")
    return value


def position_dynamics_residual(
    state0: ObjectState2D,
    state1: ObjectState2D,
) -> np.ndarray:
    """计算 r_p = p1 - (p0 + v0 * dt + 0.5 * a0 * dt**2)。

    dt 来自两个状态的时间戳，单位为秒。需要两端的位置以及前一状态的
    速度、加速度；不需要后一状态的速度或加速度。
    返回 [rx, ry]，PIXEL 下单位为 px，WORLD 下单位为 m。
    """
    dt = _time_delta(state0, state1)
    p0 = _require_vector(state0.position, "state0.position")
    p1 = _require_vector(state1.position, "state1.position")
    v0 = _require_vector(state0.velocity, "state0.velocity")
    a0 = _require_vector(state0.acceleration, "state0.acceleration")

    predicted = propagate_position(p0, v0, a0, dt)
    return p1.as_array() - predicted.as_array()


def velocity_dynamics_residual(
    state0: ObjectState2D,
    state1: ObjectState2D,
) -> np.ndarray:
    """计算 r_v = v1 - (v0 + a0 * dt)。

    需要两端的速度和前一状态的加速度。
    返回 [rx, ry]，PIXEL 下单位为 px/s，WORLD 下单位为 m/s。
    """
    dt = _time_delta(state0, state1)
    v0 = _require_vector(state0.velocity, "state0.velocity")
    v1 = _require_vector(state1.velocity, "state1.velocity")
    a0 = _require_vector(state0.acceleration, "state0.acceleration")

    predicted = propagate_velocity(v0, a0, dt)
    return v1.as_array() - predicted.as_array()


def acceleration_smoothness_residual(
    state0: ObjectState2D,
    state1: ObjectState2D,
) -> np.ndarray:
    """计算 r_a = a1 - a0，表达相邻状态的加速度应保持接近。

    需要两端的加速度，仍要求时间严格递增。
    这里使用原始加速度差，不除以 dt，因此不是 jerk（加加速度）。
    返回 [rx, ry]，PIXEL 下单位为 px/s^2，WORLD 下单位为 m/s^2。
    """
    _time_delta(state0, state1)
    a0 = _require_vector(state0.acceleration, "state0.acceleration")
    a1 = _require_vector(state1.acceleration, "state1.acceleration")
    return a1.as_array() - a0.as_array()

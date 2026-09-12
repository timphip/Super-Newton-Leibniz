"""已知状态的前向演化。使用现有匀加速传播算子，无接触、碰撞或阻力。"""

from dataclasses import replace

import numpy as np

from operators.classical import ConstantForce2D, require_world
from operators.kinematics import propagate_position, propagate_velocity
from world.schema import ObjectState2D, ObjectTrack2D


def rollout_constant_acceleration(
    initial: ObjectState2D, timestamps: list[float],
) -> ObjectTrack2D:
    """在指定时刻向前预测；时刻必须递增且不早于 initial.timestamp。"""
    require_world(initial)
    if initial.velocity is None or initial.acceleration is None:
        raise ValueError("initial velocity and acceleration are required")
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        raise ValueError("timestamps must be a nonempty finite sequence")
    if not np.isfinite(initial.timestamp) or times[0] < initial.timestamp or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must increase and not precede the initial state")
    if not np.isfinite([*initial.position.as_array(), *initial.velocity.as_array(), *initial.acceleration.as_array()]).all():
        raise ValueError("initial state vectors must be finite")
    frame_offset = 0 if times[0] == initial.timestamp else 1
    states = []
    for index, timestamp in enumerate(times):
        dt = float(timestamp - initial.timestamp)
        states.append(replace(
            initial, frame_id=initial.frame_id + index + frame_offset,
            timestamp=float(timestamp),
            position=propagate_position(initial.position, initial.velocity, initial.acceleration, dt),
            velocity=propagate_velocity(initial.velocity, initial.acceleration, dt),
            acceleration=type(initial.acceleration)(initial.acceleration.x, initial.acceleration.y),
        ))
    return ObjectTrack2D(initial.object_id, "point-mass", states)


def simulate_constant_force(
    initial: ObjectState2D, timestamps: list[float], forces: ConstantForce2D,
) -> ObjectTrack2D:
    """先由 F=ma 确定加速度，再前向演化；不从 initial.acceleration 取真值。"""
    require_world(initial)
    acceleration = forces.acceleration(initial.mass)
    return rollout_constant_acceleration(replace(initial, acceleration=acceleration), timestamps)

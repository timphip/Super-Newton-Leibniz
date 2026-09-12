"""将相邻状态的动力学残差汇总到整条二维轨迹。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constraints.dynamics import (
    acceleration_smoothness_residual,
    position_dynamics_residual,
    velocity_dynamics_residual,
)
from world.schema import ObjectTrack2D


@dataclass
class TrackDynamicsResiduals:
    """三组原始残差，各自形状为 (N - 1, 2)。

    第 i 行对应输入轨迹的 states[i] -> states[i + 1]，两列分别为 x、y。
    position 的单位为 px 或 m，velocity 为 px/s 或 m/s，
    acceleration_smoothness 为 px/s^2 或 m/s^2。
    三组量的单位不同，分别保留，不直接相加成总分。
    """

    position: np.ndarray
    velocity: np.ndarray
    acceleration_smoothness: np.ndarray


def track_dynamics_residuals(track: ObjectTrack2D) -> TrackDynamicsResiduals:
    """按存储顺序计算每一对相邻状态的三个动力学残差。

    轨迹至少需要两个状态，且都属于 track.object_id，使用同一坐标系，
    时间戳严格递增。汇总全部三种残差时，每个状态都需要速度和加速度。
    时间间隔逐段取自时间戳，允许不等间隔采样。

    本函数不排序、不修改轨迹。无效区间会抛出含状态下标和帧号的
    ValueError，不跳过坏数据。返回数组独立于输入状态。
    """
    if len(track.states) < 2:
        raise ValueError("track must contain at least two states")

    for index, state in enumerate(track.states):
        if state.object_id != track.object_id:
            raise ValueError(
                f"track {track.object_id!r}: states[{index}].object_id "
                f"is {state.object_id!r}; it must match track.object_id"
            )

    position = []
    velocity = []
    acceleration_smoothness = []

    for index, (state0, state1) in enumerate(zip(track.states, track.states[1:])):
        try:
            position.append(position_dynamics_residual(state0, state1))
            velocity.append(velocity_dynamics_residual(state0, state1))
            acceleration_smoothness.append(
                acceleration_smoothness_residual(state0, state1)
            )
        except ValueError as exc:
            raise ValueError(
                f"track {track.object_id!r}, states[{index}] -> states[{index + 1}] "
                f"(frames {state0.frame_id} -> {state1.frame_id}): {exc}"
            ) from exc

    return TrackDynamicsResiduals(
        position=np.stack(position),
        velocity=np.stack(velocity),
        acceleration_smoothness=np.stack(acceleration_smoothness),
    )

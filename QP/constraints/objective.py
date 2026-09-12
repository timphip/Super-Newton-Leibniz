"""把二维动力学残差转换为无量纲的最小二乘代价。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from constraints.trajectory import TrackDynamicsResiduals


@dataclass(frozen=True)
class DynamicsScales:
    """三个残差组的正数尺度，每组的 x、y 分量使用相同尺度。

    尺度越大，同样的残差对代价的贡献越小。必须由调用者明确提供：
    position 与位置残差同单位，velocity 与速度残差同单位，
    acceleration_smoothness 与加速度差同单位。
    它们可表示选定的容差，不自动等于观测噪声的标准差。
    """

    position: float
    velocity: float
    acceleration_smoothness: float

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration_smoothness"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} scale must be finite and greater than 0")


@dataclass
class DynamicsCost:
    """无量纲代价，均采用平方和，不带 1/2 系数。

    per_interval[i] 对应原轨迹的 states[i] -> states[i + 1]。
    mean_per_interval 是按区间数平均，不是按时间积分或平均。
    这些数衡量给定尺度下的模型一致性，不是概率或观测拟合误差。
    """

    position: float
    velocity: float
    acceleration_smoothness: float
    per_interval: np.ndarray

    @property
    def total(self) -> float:
        return float(self.per_interval.sum())

    @property
    def mean_per_interval(self) -> float:
        return self.total / len(self.per_interval)


def normalized_dynamics_residual_vector(
    residuals: TrackDynamicsResiduals,
    scales: DynamicsScales,
) -> np.ndarray:
    """返回独立的一维无量纲残差向量，保留正负号。

    每个区间的排列为 [px, py, vx, vy, ax, ay]，各分量先除以对应尺度。
    K 个区间返回 shape (6 * K,)。输入三组数组须有相同的 (K, 2) 形状，
    K >= 1，且所有分量有限。本函数不修改输入。
    """
    groups = []
    expected_shape = None
    for name in ("position", "velocity", "acceleration_smoothness"):
        values = np.asarray(getattr(residuals, name), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError(f"{name} residuals must have shape (K, 2), K >= 1")
        if expected_shape is not None and values.shape != expected_shape:
            raise ValueError("all residual groups must have the same shape")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} residuals must contain finite components")
        expected_shape = values.shape
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                groups.append(values / getattr(scales, name))
        except FloatingPointError as exc:
            raise ValueError(f"{name} normalized residuals exceed numeric range") from exc

    return np.concatenate(groups, axis=1).reshape(-1)


def dynamics_cost(
    residuals: TrackDynamicsResiduals,
    scales: DynamicsScales,
) -> DynamicsCost:
    """计算 sum(||r_p / s_p||^2 + ||r_v / s_v||^2 + ||r_a / s_a||^2)。

    每个相邻区间贡献一次，不额外乘除 dt。改变采样方式会改变这个目标。
    比较分数时需使用一致的模型、尺度约定和可比的采样方式。
    """
    normalized = normalized_dynamics_residual_vector(residuals, scales).reshape(-1, 6)
    try:
        with np.errstate(over="raise", invalid="raise"):
            squared = normalized**2
            per_interval = squared.sum(axis=1)
            # 总和也必须有限，防止各区间有限但累计后溢出。
            per_interval.sum()
            return DynamicsCost(
                position=float(squared[:, :2].sum()),
                velocity=float(squared[:, 2:4].sum()),
                acceleration_smoothness=float(squared[:, 4:6].sum()),
                per_interval=per_interval,
            )
    except FloatingPointError as exc:
        raise ValueError("dynamics cost exceeds numeric range") from exc

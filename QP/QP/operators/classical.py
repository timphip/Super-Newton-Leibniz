"""单个质点的经典动力学：恒定重力、恒定外力和能量诊断，均使用 SI 单位。"""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from world.schema import CoordinateSystem, ObjectState2D, Vec2


def require_mass(mass: float | None) -> float:
    if mass is None or not isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive, in kg")
    return mass


def require_world(state: ObjectState2D) -> None:
    if state.space != CoordinateSystem.WORLD:
        raise ValueError("classical dynamics requires WORLD coordinates in meters")


@dataclass(frozen=True)
class ConstantForce2D:
    """F_net = m*g + F_applied。g 单位 m/s²，外力单位 N，时间内保持恒定。"""

    gravity: Vec2 = field(default_factory=lambda: Vec2(0.0, -9.81))
    applied_force: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))

    def __post_init__(self) -> None:
        if not np.isfinite([*self.gravity.as_array(), *self.applied_force.as_array()]).all():
            raise ValueError("forces and gravity must be finite")

    def net_force(self, mass: float) -> Vec2:
        return Vec2.from_array(require_mass(mass) * self.gravity.as_array() + self.applied_force.as_array())

    def acceleration(self, mass: float) -> Vec2:
        return Vec2.from_array(self.net_force(mass).as_array() / require_mass(mass))


def newton_residual(state: ObjectState2D, forces: ConstantForce2D) -> np.ndarray:
    """r_F = m*a - F_net，返回 N；用于检查给定受力模型是否解释估计加速度。"""
    require_world(state)
    mass = require_mass(state.mass)
    if state.acceleration is None or not np.isfinite(state.acceleration.as_array()).all():
        raise ValueError("finite acceleration is required")
    return mass * state.acceleration.as_array() - forces.net_force(mass).as_array()


def kinetic_energy(state: ObjectState2D) -> float:
    """K = 1/2*m*|v|²，单位 J。"""
    require_world(state)
    mass = require_mass(state.mass)
    if state.velocity is None or not np.isfinite(state.velocity.as_array()).all():
        raise ValueError("finite velocity is required")
    velocity = state.velocity.as_array()
    return float(0.5 * mass * np.dot(velocity, velocity))


def gravitational_potential_energy(state: ObjectState2D, gravity: Vec2) -> float:
    """U_g = -m*g·p，以世界原点为零势能点。外力做功时 K+U_g 不必守恒。"""
    require_world(state)
    mass = require_mass(state.mass)
    if not np.isfinite([*state.position.as_array(), *gravity.as_array()]).all():
        raise ValueError("position and gravity must be finite")
    return float(-mass * np.dot(gravity.as_array(), state.position.as_array()))

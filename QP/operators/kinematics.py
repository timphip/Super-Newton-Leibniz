import numpy as np

from world.schema import Vec2, ObjectState2D


def displacement(p0: Vec2, p1: Vec2) -> Vec2:
    """
    位移：

        Δp = p1 - p0
    """

    result = p1.as_array() - p0.as_array()

    return Vec2.from_array(result)


def velocity_from_positions(
    p0: Vec2,
    p1: Vec2,
    dt: float,
) -> Vec2:
    """
    根据两个位置计算平均速度：

        v = (p1 - p0) / dt
    """

    if dt <= 0:
        raise ValueError("dt must be greater than 0")

    velocity = (
        p1.as_array() - p0.as_array()
    ) / dt

    return Vec2.from_array(velocity)


def acceleration_from_velocities(
    v0: Vec2,
    v1: Vec2,
    dt: float,
) -> Vec2:
    """
    根据两个速度计算平均加速度：

        a = (v1 - v0) / dt
    """

    if dt <= 0:
        raise ValueError("dt must be greater than 0")

    acceleration = (
        v1.as_array() - v0.as_array()
    ) / dt

    return Vec2.from_array(acceleration)


def propagate_position(
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
    dt: float,
) -> Vec2:
    """
    匀加速情况下传播位置：

        p(t+dt)
        =
        p(t)
        + v(t) dt
        + 1/2 a(t) dt²
    """

    if dt < 0:
        raise ValueError("dt must be non-negative")

    p = position.as_array()
    v = velocity.as_array()
    a = acceleration.as_array()

    next_position = (
        p
        + v * dt
        + 0.5 * a * dt**2
    )

    return Vec2.from_array(next_position)


def propagate_velocity(
    velocity: Vec2,
    acceleration: Vec2,
    dt: float,
) -> Vec2:
    """
    匀加速情况下传播速度：

        v(t+dt)
        =
        v(t)
        + a(t) dt
    """

    if dt < 0:
        raise ValueError("dt must be non-negative")

    v = velocity.as_array()
    a = acceleration.as_array()

    next_velocity = v + a * dt

    return Vec2.from_array(next_velocity)


def speed(velocity: Vec2) -> float:
    """
    速度大小：

        |v| = sqrt(vx² + vy²)
    """

    return velocity.norm()


def acceleration_magnitude(acceleration: Vec2) -> float:
    """
    加速度大小：

        |a| = sqrt(ax² + ay²)
    """

    return acceleration.norm()
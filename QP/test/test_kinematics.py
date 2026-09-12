from world.schema import Vec2
from operators.kinematics import (
    displacement,
    velocity_from_positions,
    acceleration_from_velocities,
    propagate_position,
    propagate_velocity,
)


def test_displacement():
    p0 = Vec2(1.0, 2.0)
    p1 = Vec2(4.0, 6.0)

    dp = displacement(p0, p1)

    assert dp.x == 3.0
    assert dp.y == 4.0


def test_velocity():
    p0 = Vec2(0.0, 0.0)
    p1 = Vec2(10.0, 4.0)

    v = velocity_from_positions(
        p0,
        p1,
        dt=2.0,
    )

    assert v.x == 5.0
    assert v.y == 2.0


def test_acceleration():
    v0 = Vec2(10.0, 0.0)
    v1 = Vec2(14.0, 6.0)

    a = acceleration_from_velocities(
        v0,
        v1,
        dt=2.0,
    )

    assert a.x == 2.0
    assert a.y == 3.0


def test_state_propagation():
    p0 = Vec2(0.0, 0.0)
    v0 = Vec2(10.0, 0.0)
    a0 = Vec2(2.0, 0.0)

    dt = 1.0

    p1 = propagate_position(
        p0,
        v0,
        a0,
        dt,
    )

    v1 = propagate_velocity(
        v0,
        a0,
        dt,
    )

    assert p1.x == 11.0
    assert p1.y == 0.0

    assert v1.x == 12.0
    assert v1.y == 0.0
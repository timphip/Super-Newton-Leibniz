from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from constraints.dynamics import (
    acceleration_smoothness_residual,
    position_dynamics_residual,
    velocity_dynamics_residual,
)
from world.schema import CoordinateSystem, ObjectState2D, Vec2


RESIDUALS = (
    position_dynamics_residual,
    velocity_dynamics_residual,
    acceleration_smoothness_residual,
)


@pytest.fixture
def states():
    # 已手算的二维匀加速轨迹，dt = 0.5 s；包含负方向运动。
    state0 = ObjectState2D(
        object_id="ball",
        frame_id=10,
        timestamp=10.25,
        space=CoordinateSystem.WORLD,
        position=Vec2(1.0, -2.0),
        velocity=Vec2(3.0, -4.0),
        acceleration=Vec2(2.0, 6.0),
    )
    state1 = ObjectState2D(
        object_id="ball",
        frame_id=25,
        timestamp=10.75,
        space=CoordinateSystem.WORLD,
        position=Vec2(2.75, -3.25),
        velocity=Vec2(4.0, -1.0),
        acceleration=Vec2(2.0, 6.0),
    )
    return state0, state1


@pytest.mark.parametrize("residual", RESIDUALS)
@pytest.mark.parametrize("space", list(CoordinateSystem))
@pytest.mark.parametrize(
    "timestamp, position, velocity",
    [
        (10.75, Vec2(2.75, -3.25), Vec2(4.0, -1.0)),
        (12.25, Vec2(11.0, 2.0), Vec2(7.0, 8.0)),
    ],
)
def test_constant_acceleration_has_zero_residual(
    states, residual, space, timestamp, position, velocity
):
    state0, state1 = states
    state0 = replace(state0, space=space)
    state1 = replace(
        state1, space=space, timestamp=timestamp,
        position=position, velocity=velocity,
    )

    result = residual(state0, state1)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "residual, field, value, expected",
    [
        (position_dynamics_residual, "position", Vec2(3.25, -4.75), [0.5, -1.5]),
        (velocity_dynamics_residual, "velocity", Vec2(3.0, 1.0), [-1.0, 2.0]),
        (acceleration_smoothness_residual, "acceleration", Vec2(-1.0, 8.0), [-3.0, 2.0]),
    ],
)
def test_residual_preserves_signed_error(states, residual, field, value, expected):
    state0, state1 = states
    state1 = replace(state1, **{field: value})

    np.testing.assert_allclose(residual(state0, state1), expected, atol=1e-12)


def test_smoothness_is_acceleration_difference_not_jerk(states):
    state0, state1 = states
    state1 = replace(state1, acceleration=Vec2(3.0, 4.0))

    for timestamp in (10.75, 12.25):
        result = acceleration_smoothness_residual(
            state0, replace(state1, timestamp=timestamp)
        )
        np.testing.assert_allclose(result, [1.0, -2.0], atol=1e-12)


@pytest.mark.parametrize("residual", RESIDUALS)
@pytest.mark.parametrize("timestamp", [10.25, 10.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_end_timestamp_is_rejected(states, residual, timestamp):
    state0, state1 = states
    with pytest.raises(ValueError, match="timestamp"):
        residual(state0, replace(state1, timestamp=timestamp))


@pytest.mark.parametrize("residual", RESIDUALS)
@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_start_timestamp_is_rejected(states, residual, timestamp):
    state0, state1 = states
    with pytest.raises(ValueError, match="timestamp"):
        residual(replace(state0, timestamp=timestamp), state1)


@pytest.mark.parametrize("residual", RESIDUALS)
def test_overflowing_time_interval_is_rejected(states, residual):
    state0, state1 = states
    with pytest.raises(ValueError, match="timestamp"):
        residual(replace(state0, timestamp=-1e308), replace(state1, timestamp=1e308))


@pytest.mark.parametrize("residual", RESIDUALS)
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"object_id": "another-ball"}, "object_id"),
        ({"space": CoordinateSystem.PIXEL}, "coordinate system"),
    ],
)
def test_incompatible_states_are_rejected(states, residual, changes, message):
    state0, state1 = states
    with pytest.raises(ValueError, match=message):
        residual(state0, replace(state1, **changes))


@pytest.mark.parametrize(
    "residual, index, field",
    [
        (position_dynamics_residual, 0, "position"),
        (position_dynamics_residual, 1, "position"),
        (position_dynamics_residual, 0, "velocity"),
        (position_dynamics_residual, 0, "acceleration"),
        (velocity_dynamics_residual, 0, "velocity"),
        (velocity_dynamics_residual, 1, "velocity"),
        (velocity_dynamics_residual, 0, "acceleration"),
        (acceleration_smoothness_residual, 0, "acceleration"),
        (acceleration_smoothness_residual, 1, "acceleration"),
    ],
)
@pytest.mark.parametrize("value", [None, Vec2(float("nan"), 0.0), Vec2(0.0, float("inf"))])
def test_required_vectors_must_exist_and_be_finite(states, residual, index, field, value):
    pair = list(states)
    pair[index] = replace(pair[index], **{field: value})

    with pytest.raises(ValueError, match=f"state{index}[.]{field}"):
        residual(*pair)


@pytest.mark.parametrize(
    "residual, changes0, changes1",
    [
        (position_dynamics_residual, {}, {"velocity": None, "acceleration": None}),
        (velocity_dynamics_residual, {}, {"acceleration": None}),
        (acceleration_smoothness_residual, {"velocity": None}, {"velocity": None}),
    ],
)
def test_unused_optional_fields_may_be_missing(states, residual, changes0, changes1):
    state0, state1 = states
    result = residual(replace(state0, **changes0), replace(state1, **changes1))
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("residual", RESIDUALS)
def test_residual_does_not_modify_input_states(states, residual):
    before = deepcopy(states)
    result = residual(*states)
    result[:] = 999.0

    assert states == before

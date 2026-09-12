from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from constraints.objective import (
    DynamicsScales,
    dynamics_cost,
    normalized_dynamics_residual_vector,
)
from constraints.trajectory import TrackDynamicsResiduals, track_dynamics_residuals
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, Vec2


@pytest.fixture
def residuals():
    return TrackDynamicsResiduals(
        position=np.array([[2.0, -4.0], [0.0, 2.0]]),
        velocity=np.array([[3.0, 0.0], [-3.0, 6.0]]),
        acceleration_smoothness=np.array([[0.0, 4.0], [4.0, -4.0]]),
    )


@pytest.fixture
def scales():
    return DynamicsScales(position=2.0, velocity=3.0, acceleration_smoothness=4.0)


def test_normalized_vector_preserves_sign_and_interval_order(residuals, scales):
    result = normalized_dynamics_residual_vector(residuals, scales)

    assert result.shape == (12,)
    np.testing.assert_allclose(
        result, [1.0, -2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 2.0, 1.0, -1.0],
    )


def test_hand_calculated_cost_breakdown(residuals, scales):
    result = dynamics_cost(residuals, scales)

    assert result.position == pytest.approx(6.0)
    assert result.velocity == pytest.approx(6.0)
    assert result.acceleration_smoothness == pytest.approx(3.0)
    np.testing.assert_allclose(result.per_interval, [7.0, 8.0])
    assert result.total == pytest.approx(15.0)
    assert result.mean_per_interval == pytest.approx(7.5)


def test_doubling_position_scale_quarters_only_position_cost(residuals, scales):
    original = dynamics_cost(residuals, scales)
    relaxed = dynamics_cost(residuals, replace(scales, position=4.0))

    assert relaxed.position == pytest.approx(original.position / 4.0)
    assert relaxed.velocity == original.velocity
    assert relaxed.acceleration_smoothness == original.acceleration_smoothness


def test_consistent_length_unit_conversion_preserves_cost(residuals, scales):
    # 长度从米改为厘米，位置、速度、加速度及对应尺度都乘以 100。
    converted = TrackDynamicsResiduals(
        position=residuals.position * 100.0,
        velocity=residuals.velocity * 100.0,
        acceleration_smoothness=residuals.acceleration_smoothness * 100.0,
    )
    converted_scales = DynamicsScales(200.0, 300.0, 400.0)
    original = dynamics_cost(residuals, scales)
    result = dynamics_cost(converted, converted_scales)

    np.testing.assert_allclose(result.per_interval, original.per_interval)
    assert result.total == pytest.approx(original.total)


def test_full_track_pipeline_detects_a_position_error():
    state0 = ObjectState2D(
        object_id="ball", frame_id=0, timestamp=0.0,
        space=CoordinateSystem.WORLD, position=Vec2(0.0, 0.0),
        velocity=Vec2(1.0, -2.0), acceleration=Vec2(2.0, 4.0),
    )
    state1 = replace(
        state0, frame_id=15, timestamp=0.5,
        position=Vec2(0.75, -0.5), velocity=Vec2(2.0, 0.0),
    )
    track = ObjectTrack2D("ball", "ball", [state0, state1])
    scales = DynamicsScales(0.1, 0.2, 0.5)

    assert dynamics_cost(track_dynamics_residuals(track), scales).total == 0.0

    track.states[1] = replace(state1, position=Vec2(0.95, -0.8))
    result = dynamics_cost(track_dynamics_residuals(track), scales)
    assert result.total == pytest.approx(13.0)
    assert result.position == pytest.approx(13.0)
    assert result.velocity == 0.0
    assert result.acceleration_smoothness == 0.0


def test_scales_must_be_positive_and_finite(scales):
    for field in ("position", "velocity", "acceleration_smoothness"):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=f"{field} scale"):
                replace(scales, **{field: value})


def test_invalid_residual_arrays_are_rejected(residuals, scales):
    for field in ("position", "velocity", "acceleration_smoothness"):
        for values in (
            np.array([1.0, 2.0]), np.empty((0, 2)), np.zeros((2, 3)),
            np.zeros((1, 2)), np.array([[float("nan"), 0.0], [0.0, 0.0]]),
            np.array([[0.0, float("inf")], [0.0, 0.0]]),
        ):
            with pytest.raises(ValueError, match="shape|finite"):
                dynamics_cost(replace(residuals, **{field: values}), scales)


def test_numeric_overflow_is_reported(residuals, scales):
    for magnitude, position_scale in ((1e308, 0.01), (1e200, 1.0), (8e153, 1.0)):
        # 分别覆盖归一化、平方、累计求和的溢出。
        huge = replace(residuals, position=np.full((2, 2), magnitude))
        with pytest.raises(ValueError, match="numeric range"):
            dynamics_cost(huge, replace(scales, position=position_scale))


def test_outputs_are_independent_of_input_arrays(residuals, scales):
    before = deepcopy(residuals)
    vector = normalized_dynamics_residual_vector(residuals, scales)
    result = dynamics_cost(residuals, scales)
    vector[:] = 999.0
    result.per_interval[:] = 999.0

    for field in ("position", "velocity", "acceleration_smoothness"):
        np.testing.assert_array_equal(getattr(residuals, field), getattr(before, field))

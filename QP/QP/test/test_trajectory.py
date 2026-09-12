from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from constraints.trajectory import track_dynamics_residuals
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, Vec2


@pytest.fixture
def track():
    # 手算的二维匀加速轨迹，两个时间间隔分别为 0.5 s 和 1.5 s。
    return ObjectTrack2D(
        object_id="ball",
        label="ball",
        states=[
            ObjectState2D(
                object_id="ball", frame_id=0, timestamp=0.0,
                space=CoordinateSystem.WORLD,
                position=Vec2(0.0, 0.0), velocity=Vec2(1.0, -2.0),
                acceleration=Vec2(2.0, 4.0),
            ),
            ObjectState2D(
                object_id="ball", frame_id=15, timestamp=0.5,
                space=CoordinateSystem.WORLD,
                position=Vec2(0.75, -0.5), velocity=Vec2(2.0, 0.0),
                acceleration=Vec2(2.0, 4.0),
            ),
            ObjectState2D(
                object_id="ball", frame_id=60, timestamp=2.0,
                space=CoordinateSystem.WORLD,
                position=Vec2(6.0, 4.0), velocity=Vec2(5.0, 6.0),
                acceleration=Vec2(2.0, 4.0),
            ),
        ],
    )


def test_constant_acceleration_track_with_unequal_intervals(track):
    result = track_dynamics_residuals(track)

    for values in (result.position, result.velocity, result.acceleration_smoothness):
        assert values.shape == (2, 2)
        np.testing.assert_allclose(values, np.zeros((2, 2)), atol=1e-12)

    # 两个状态也应可用，输出必须保留“区间、方向”两个维度。
    pair_result = track_dynamics_residuals(replace(track, states=track.states[:2]))
    assert pair_result.position.shape == (1, 2)
    assert pair_result.velocity.shape == (1, 2)
    assert pair_result.acceleration_smoothness.shape == (1, 2)


def test_middle_state_error_affects_both_neighboring_intervals(track):
    track.states[1] = replace(
        track.states[1], position=Vec2(0.95, -0.8),
        velocity=Vec2(3.0, -2.0), acceleration=Vec2(3.0, 2.0),
    )

    result = track_dynamics_residuals(track)

    np.testing.assert_allclose(
        result.position, [[0.2, -0.3], [-2.825, 5.55]], atol=1e-12,
    )
    np.testing.assert_allclose(
        result.velocity, [[1.0, -2.0], [-2.5, 5.0]], atol=1e-12,
    )
    np.testing.assert_allclose(
        result.acceleration_smoothness, [[1.0, -2.0], [-1.0, 2.0]], atol=1e-12,
    )


def test_track_needs_at_least_two_states(track):
    for states in ([], track.states[:1]):
        with pytest.raises(ValueError, match="at least two states"):
            track_dynamics_residuals(replace(track, states=states))


def test_state_ids_must_match_the_track_id(track):
    # 状态之间的 ID 相同，但仍必须与所属轨迹一致。
    with pytest.raises(ValueError, match=r"states\[0\].object_id.*track.object_id"):
        track_dynamics_residuals(replace(track, object_id="another-ball"))


def test_invalid_later_interval_reports_location_without_sorting(track):
    track.states[2] = replace(track.states[2], timestamp=0.25)
    before = deepcopy(track)

    with pytest.raises(ValueError) as error:
        track_dynamics_residuals(track)

    message = str(error.value)
    assert "states[1] -> states[2]" in message
    assert "frames 15 -> 60" in message
    assert "timestamp" in message
    assert track == before


def test_missing_final_acceleration_reports_the_affected_interval(track):
    track.states[2] = replace(track.states[2], acceleration=None)

    with pytest.raises(ValueError) as error:
        track_dynamics_residuals(track)

    message = str(error.value)
    assert "states[1] -> states[2]" in message
    assert "state1.acceleration is required" in message


def test_returned_arrays_do_not_modify_the_track(track):
    before = deepcopy(track)
    result = track_dynamics_residuals(track)

    result.position[:] = 999.0
    result.velocity[:] = 999.0
    result.acceleration_smoothness[:] = 999.0

    assert track == before

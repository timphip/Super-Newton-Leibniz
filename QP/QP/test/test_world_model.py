from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from constraints.observation import observation_residual_vector, position_observation_residual
from examples.world_model_demo import make_case
from inference.constant_acceleration import fit_constant_acceleration
from operators.classical import (
    ConstantForce2D, gravitational_potential_energy, kinetic_energy, newton_residual,
)
from world.calibration import PlanarCalibration2D
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, PixelObservation2D, Vec2
from world.simulation import rollout_constant_acceleration, simulate_constant_force


@pytest.fixture
def calibration():
    return PlanarCalibration2D(Vec2(100.0, 500.0), 0.02)


@pytest.fixture
def initial():
    return ObjectState2D(
        "ball", 0, 0.0, CoordinateSystem.WORLD, Vec2(0.0, 10.0),
        velocity=Vec2(3.0, 4.0), mass=2.0,
    )


def observe(track, calibration):
    return [PixelObservation2D(
        s.object_id, s.frame_id, s.timestamp, calibration.to_pixel(s.position), sigma_px=1.0,
    ) for s in track.states]


def test_calibration_handles_pixel_axis_direction_and_round_trip(calibration):
    assert calibration.to_world(Vec2(150.0, 400.0)) == Vec2(1.0, 2.0)
    assert calibration.to_pixel(Vec2(1.0, 2.0)) == Vec2(150.0, 400.0)
    for scale in (0.0, -0.1, float("inf")):
        with pytest.raises(ValueError, match="meters_per_pixel"):
            PlanarCalibration2D(Vec2(0, 0), scale)


def test_observation_residual_is_projected_state_minus_measurement(initial, calibration):
    observation = PixelObservation2D("ball", 0, 0.0, Vec2(98.0, 4.0), sigma_px=2.0)
    np.testing.assert_allclose(position_observation_residual(initial, observation, calibration), [2.0, -4.0])
    track = ObjectTrack2D("ball", "ball", [initial])
    np.testing.assert_allclose(observation_residual_vector(track, [observation], calibration), [1.0, -2.0])
    pixel_state = replace(initial, space=CoordinateSystem.PIXEL, position=Vec2(100.0, 0.0))
    np.testing.assert_allclose(position_observation_residual(pixel_state, observation), [2.0, -4.0])


def test_observation_matching_and_world_calibration_are_required(initial, calibration):
    observation = PixelObservation2D("ball", 0, 0.0, Vec2(100.0, 0.0))
    with pytest.raises(ValueError, match="calibration"):
        position_observation_residual(initial, observation)
    for changes in ({"object_id": "other"}, {"frame_id": 1}, {"timestamp": 0.1}, {"sigma_px": 0.0}):
        with pytest.raises(ValueError):
            position_observation_residual(initial, replace(observation, **changes), calibration)


def test_gravity_and_external_force_follow_newtons_second_law():
    forces = ConstantForce2D(Vec2(0.0, -10.0), Vec2(6.0, 0.0))
    assert forces.net_force(2.0) == Vec2(6.0, -20.0)
    assert forces.acceleration(2.0) == Vec2(3.0, -10.0)
    assert forces.acceleration(4.0) == Vec2(1.5, -10.0)


def test_projectile_matches_analytic_solution_and_conserves_energy(initial):
    forces = ConstantForce2D(Vec2(0.0, -10.0))
    before = deepcopy(initial)
    track = simulate_constant_force(initial, [0.0, 0.5, 1.0], forces)
    end = track.states[-1]
    assert end.position == Vec2(3.0, 9.0)
    assert end.velocity == Vec2(3.0, -6.0)
    energies = [kinetic_energy(s) + gravitational_potential_energy(s, forces.gravity) for s in track.states]
    np.testing.assert_allclose(energies, [225.0] * 3, atol=1e-10)
    np.testing.assert_allclose(newton_residual(end, forces), [0.0, 0.0])
    assert initial == before


def test_external_force_work_matches_energy_change(initial):
    forces = ConstantForce2D(Vec2(0.0, 0.0), Vec2(4.0, 0.0))
    track = simulate_constant_force(initial, [0.0, 1.0], forces)
    start, end = track.states
    work = np.dot(forces.applied_force.as_array(), end.position.as_array() - start.position.as_array())
    assert kinetic_energy(end) - kinetic_energy(start) == pytest.approx(work)
    incorrect = replace(end, acceleration=Vec2(3.0, 0.0))
    np.testing.assert_allclose(newton_residual(incorrect, forces), [2.0, 0.0])


def test_classical_dynamics_requires_mass_and_metric_coordinates(initial):
    for mass in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="mass"):
            simulate_constant_force(replace(initial, mass=mass), [0.0, 1.0], ConstantForce2D())
    with pytest.raises(ValueError, match="WORLD"):
        simulate_constant_force(replace(initial, space=CoordinateSystem.PIXEL), [0.0, 1.0], ConstantForce2D())


def test_fit_recovers_states_from_irregularly_sampled_observations(initial, calibration):
    initial = replace(initial, timestamp=1e9)
    truth = simulate_constant_force(initial, [1e9, 1e9 + 0.25, 1e9 + 0.75, 1e9 + 1.5], ConstantForce2D())
    observations = observe(truth, calibration)
    before = deepcopy(observations)
    result = fit_constant_acceleration(observations, calibration, mass=2.0)
    for expected, actual in zip(truth.states, result.states):
        for field in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(actual, field).as_array(), getattr(expected, field).as_array(), atol=1e-10)
        assert actual.timestamp == expected.timestamp
        assert actual.frame_id == expected.frame_id
    assert observations == before


def test_known_acceleration_needs_only_two_observations(initial, calibration):
    truth = simulate_constant_force(initial, [0.0, 0.5], ConstantForce2D())
    result = fit_constant_acceleration(observe(truth, calibration), calibration, known_acceleration=Vec2(0.0, -9.81))
    np.testing.assert_allclose(result.states[0].velocity.as_array(), [3.0, 4.0], atol=1e-10)
    assert result.states[0].acceleration == Vec2(0.0, -9.81)


def test_uncertain_observation_has_less_influence(initial, calibration):
    truth = simulate_constant_force(initial, [0.0, 0.25, 0.5, 0.75, 1.0], ConstantForce2D())
    observations = observe(truth, calibration)
    observations[2] = replace(observations[2], center=Vec2(observations[2].center.x + 100, observations[2].center.y))
    unweighted = fit_constant_acceleration(observations, calibration)
    observations[2] = replace(observations[2], sigma_px=1000.0)
    weighted = fit_constant_acceleration(observations, calibration)
    assert abs(weighted.states[0].acceleration.x) < abs(unweighted.states[0].acceleration.x) * 0.01


def test_fit_rejects_ambiguous_or_invalid_observation_sequences(initial, calibration):
    truth = simulate_constant_force(initial, [0.0, 0.5, 1.0], ConstantForce2D())
    observations = observe(truth, calibration)
    for invalid in (
        observations[:2], list(reversed(observations)),
        [observations[0], observations[0], observations[2]],
        [observations[0], replace(observations[1], object_id="other"), observations[2]],
    ):
        with pytest.raises(ValueError):
            fit_constant_acceleration(invalid, calibration)


def test_forecast_uses_last_estimate_without_future_observations(initial, calibration):
    truth = simulate_constant_force(initial, [0.0, 0.25, 0.5], ConstantForce2D())
    estimated = fit_constant_acceleration(observe(truth, calibration), calibration, mass=2.0)
    future = rollout_constant_acceleration(estimated.states[-1], [0.75, 1.0])
    np.testing.assert_allclose(future.states[-1].position.as_array(), [3.0, 9.095], atol=1e-10)
    assert future.states[0].frame_id == 3
    with pytest.raises(ValueError, match="timestamps"):
        rollout_constant_acceleration(estimated.states[-1], [0.25])


def test_synthetic_demo_separates_fitting_and_future_evaluation():
    case = make_case("projectile", 0.0, False)
    assert case["metrics"]["future_rmse_m"] < 1e-10
    assert all(row["observed"] is None for row in case["rows"][case["fit_count"]:])
    assert all(row["observed"] is not None for row in case["rows"][:case["fit_count"]])
    noisy = make_case("projectile", 6.0, False)
    assert noisy["metrics"]["future_rmse_m"] > 0.0
    assert noisy == make_case("projectile", 6.0, False)

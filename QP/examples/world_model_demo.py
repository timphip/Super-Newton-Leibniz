"""合成观测 -> 状态估计 -> 经典动力学诊断 -> 保留未来时段预测。

运行 python -m examples.world_model_demo --output <json路径>。
合成真值只用于生成带噪观测、事后评测和显示，不传入估计器。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from constraints.objective import DynamicsScales, dynamics_cost
from constraints.observation import observation_residual_vector
from constraints.trajectory import track_dynamics_residuals
from inference.constant_acceleration import fit_constant_acceleration
from operators.classical import ConstantForce2D, gravitational_potential_energy, kinetic_energy, newton_residual
from world.calibration import PlanarCalibration2D
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, PixelObservation2D, Vec2
from world.simulation import rollout_constant_acceleration, simulate_constant_force


SCENARIOS = {
    "projectile": ("抛体运动", Vec2(0.0, 1.5), Vec2(4.0, 9.0), ConstantForce2D(), 1.8),
    "fall": ("自由落体", Vec2(2.0, 20.0), Vec2(0.0, 0.0), ConstantForce2D(), 1.8),
    "force": ("恒定水平外力", Vec2(0.0, 1.0), Vec2(1.0, 0.0), ConstantForce2D(Vec2(0.0, 0.0), Vec2(3.0, 0.0)), 3.0),
}


def make_case(scenario: str, noise_px: float, use_known_force: bool) -> dict:
    label, position, velocity, forces, duration = SCENARIOS[scenario]
    initial = ObjectState2D(
        "body", 0, 0.0, CoordinateSystem.WORLD, position,
        velocity=velocity, mass=2.0,
    )
    times = np.linspace(0.0, duration, 61).tolist()
    truth = simulate_constant_force(initial, times, forces)
    calibration = PlanarCalibration2D(Vec2(40.0, 500.0), 0.02)
    rng = np.random.default_rng(20260909)
    # 零噪声例仍用 1 px 作为评分尺度，避免 sigma=0。
    sigma_px = max(noise_px, 1.0)
    observations = [PixelObservation2D(
        "body", state.frame_id, state.timestamp,
        Vec2.from_array(calibration.to_pixel(state.position).as_array() + rng.normal(0.0, noise_px, 2)),
        sigma_px=sigma_px,
    ) for state in truth.states]

    fit_count = 37
    fit_observations = observations[:fit_count]
    fit = fit_constant_acceleration(
        fit_observations, calibration, mass=initial.mass,
        known_acceleration=forces.acceleration(initial.mass) if use_known_force else None,
    )
    future = rollout_constant_acceleration(fit.states[-1], times[fit_count:])
    estimated = ObjectTrack2D("body", "estimated", fit.states + future.states)
    measured = np.array([calibration.to_world(o.center).as_array() for o in fit_observations])
    actual = np.array([s.position.as_array() for s in truth.states])
    predicted = np.array([s.position.as_array() for s in estimated.states])
    pixel_residuals = observation_residual_vector(fit, fit_observations, calibration) * sigma_px
    position_error = np.linalg.norm(predicted - actual, axis=1)
    cost = dynamics_cost(track_dynamics_residuals(fit), DynamicsScales(0.05, 0.1, 0.5))
    sample_rows = []
    for index, (true, estimate) in enumerate(zip(truth.states, estimated.states)):
        true_energy = kinetic_energy(true) + gravitational_potential_energy(true, forces.gravity)
        estimate_energy = kinetic_energy(estimate) + gravitational_potential_energy(estimate, forces.gravity)
        sample_rows.append({
            "t": true.timestamp, "p": estimate.position.as_array().tolist(),
            "truth": true.position.as_array().tolist(), "v": estimate.velocity.as_array().tolist(),
            "a": estimate.acceleration.as_array().tolist(),
            "observed": measured[index].tolist() if index < fit_count else None,
            "energy": estimate_energy, "truth_energy": true_energy,
            "position_error": float(position_error[index]),
        })
    return {
        "scenario": scenario, "label": label, "noise_px": noise_px,
        "known_force": use_known_force, "fit_count": fit_count, "mass": initial.mass,
        "force": forces.net_force(initial.mass).as_array().tolist(),
        "gravity": forces.gravity.as_array().tolist(),
        "sigma_px": sigma_px, "meters_per_pixel": calibration.meters_per_pixel,
        "metrics": {
            "observation_rmse_px": float(np.sqrt(np.mean(pixel_residuals**2))),
            "future_rmse_m": float(np.sqrt(np.mean(position_error[fit_count:]**2))),
            "fit_rmse_m": float(np.sqrt(np.mean(position_error[:fit_count]**2))),
            "newton_residual_n": float(np.linalg.norm(newton_residual(fit.states[0], forces))),
            "dynamics_cost": cost.total,
            "observation_cost": float(np.sum((pixel_residuals / sigma_px)**2)),
        },
        "rows": sample_rows,
    }


def build_demo() -> dict:
    return {
        "schema_version": 1, "source": "synthetic", "seed": 20260909,
        "cases": [make_case(scenario, noise, known) for scenario in SCENARIOS
                  for noise in (0.0, 2.0, 6.0) for known in (False, True)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for case in data["cases"]:
        print(f"{case['scenario']:10} noise={case['noise_px']:.0f}px known_force={case['known_force']} "
              f"future_RMSE={case['metrics']['future_rmse_m']:.6f}m")


if __name__ == "__main__":
    main()

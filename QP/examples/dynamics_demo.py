"""运行：python -m examples.dynamics_demo。示例尺度仅用于演示。"""

from copy import deepcopy

from constraints.objective import DynamicsScales, dynamics_cost
from constraints.trajectory import track_dynamics_residuals
from world.schema import CoordinateSystem, ObjectState2D, ObjectTrack2D, Vec2


def make_demo_track() -> ObjectTrack2D:
    """创建三个手算状态，时间间隔分别为 0.5 s 和 1.5 s。"""
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


def main() -> None:
    scales = DynamicsScales(
        position=0.1, velocity=0.2, acceleration_smoothness=0.5,
    )
    clean = make_demo_track()
    shifted = deepcopy(clean)
    shifted.states[1].position = Vec2(0.95, -0.8)

    print("Demo scales: position=0.1 m, velocity=0.2 m/s, acceleration change=0.5 m/s^2")
    for label, track in (
        ("Constant acceleration", clean),
        ("Middle position shifted by (+0.20 m, -0.30 m)", shifted),
    ):
        residuals = track_dynamics_residuals(track)
        cost = dynamics_cost(residuals, scales)
        print(f"\n{label}")
        print(f"Position residuals (m):\n{residuals.position}")
        print(f"Position cost: {cost.position:.6f}")
        print(f"Velocity cost: {cost.velocity:.6f}")
        print(f"Acceleration smoothness cost: {cost.acceleration_smoothness:.6f}")
        for index, value in enumerate(cost.per_interval):
            state0, state1 = track.states[index:index + 2]
            print(f"Frames {state0.frame_id} -> {state1.frame_id}: {value:.6f}")
        print(f"Total: {cost.total:.6f}; mean per interval: {cost.mean_per_interval:.6f}")


if __name__ == "__main__":
    main()

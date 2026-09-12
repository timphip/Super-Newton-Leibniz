from __future__ import annotations
from dataclasses import dataclass,field
from enum import Enum
from typing import Optional
import numpy as np

##坐标系
class CoordinateSystem(str,Enum):
    """
    PIXEL:
        position -> px
        velocity -> px/s
        acceleration -> px/s^2
    WORLD:
        position -> m
        velocity -> m/s
        acceleration -> m/s^2
    """
    PIXEL = "pixel"
    WORLD = "world"

##2维度向量
@dataclass
class Vec2:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y],dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vec2":
        value = np.asarray(arr, dtype=float).reshape(2)
        return cls(x=float(value[0]), y=float(value[1]))

    def norm(self) -> float:
        return float(np.hypot(self.x, self.y))

##2D OBJ Size
@dataclass
class Size2D:
    width: float
    height: float

    def as_array(self) -> np.ndarray:
        return np.array([self.width, self.height],dtype=float)

@classmethod
def from_array(cls, arr: np.ndarray) -> "Size2D":
    value = np.asarray(arr, dtype=float).reshape(2)

    return cls(
        width=float(value[0]),
        height=float(value[1]),
    )

## State of OBJ (instant)
@dataclass
class ObjectState2D:
    """
    一个物体在某一个时间点的物理状态。

    注意：
    这里存的是 state，而不是 detector 的原始 observation。
    """

    object_id: str
    frame_id: int
    timestamp: float

    space: CoordinateSystem

    position: Vec2

    velocity: Optional[Vec2] = None
    acceleration: Optional[Vec2] = None

    size: Optional[Size2D] = None

    # 后期做动力学/能量时使用
    mass: Optional[float] = None

    # 物体二维朝向，单位 rad
    orientation: Optional[float] = None

    # 对当前 state 的整体置信度
    confidence: float = 1.0

#Raw observation from detector
@dataclass
class PixelObservation2D:
    """
    detector 的原始 observation。
    """

    object_id: str
    frame_id: int
    timestamp: float

    center: Vec2
    
    # bounding box:
    # [xmin, ymin, xmax, ymax]
    bbox: Optional[np.ndarray] = None

    # segmentation mask 可选
    mask: Optional[np.ndarray] = None

    # detector / tracker confidence
    confidence: float = 1.0

    # pixel measurement uncertainty
    sigma_px: float = 1.0

## Track
@dataclass
class ObjectTrack2D:
    """
    一个物体的轨迹。
    """

    object_id: str
    label: str
    states: list[ObjectState2D] = field(default_factory=list)

    def add_state(self, state: ObjectState2D)-> None:
        if state.object_id != self.object_id:
            raise ValueError(f"State object_id {state.object_id} does not match track object_id {self.object_id}")
        self.states.append(state)
    def sort_by_time(self) -> None:
            self.states.sort(key=lambda s: s.timestamp)

##2D scene
@dataclass
class Scene2D:
    """
    一个二维场景。
    """

    scene_id: str

    fps: float
    width_px: int
    height_px: int
    tracks: dict[str, ObjectTrack2D] = field(default_factory=dict)

    def add_track(self, track: ObjectTrack2D) -> None:
        self.tracks[track.object_id] = track
    def get_track(self, object_id: str) -> ObjectTrack2D:
        if object_id not in self.tracks:
            raise ValueError(f"Track with object_id {object_id} not found in scene {self.scene_id}")
        return self.tracks[object_id]

##Metric calibration
@dataclass
class MetricScale2D:
    gamma: float
    sigma: Optional[float] = None
    confidence: float = 1.0

##task context

@dataclass
class QuestionContext2D:
    """
    QuantiPhy 一道题对应的信息。

    Scene 本身不应该被某一道题的 prior 污染，
    因此 prior 单独放在 QuestionContext。
    """

    question_id: str

    target_object_id: str
    target_quantity: str
    # "size"
    # "distance"
    # "velocity"
    # "acceleration"

    target_time: Optional[float] = None

    prior_object_id: Optional[str] = None
    prior_quantity: Optional[str] = None
    prior_value: Optional[float] = None
    prior_unit: Optional[str] = None
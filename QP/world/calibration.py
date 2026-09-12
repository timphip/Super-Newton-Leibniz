"""固定相机、同一运动平面下的二维尺度标定；世界 y 向上，像素 y 向下。"""

from dataclasses import dataclass
from math import isfinite

from world.schema import Vec2


@dataclass(frozen=True)
class PlanarCalibration2D:
    """像素原点位置及米/像素尺度。不处理透视、相机运动或深度变化。"""

    origin_px: Vec2
    meters_per_pixel: float

    def __post_init__(self) -> None:
        if not isfinite(self.meters_per_pixel) or self.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be finite and positive")
        if not all(isfinite(v) for v in (self.origin_px.x, self.origin_px.y)):
            raise ValueError("origin_px must be finite")

    def to_world(self, pixel: Vec2) -> Vec2:
        if not all(isfinite(v) for v in (pixel.x, pixel.y)):
            raise ValueError("pixel position must be finite")
        return Vec2(
            (pixel.x - self.origin_px.x) * self.meters_per_pixel,
            (self.origin_px.y - pixel.y) * self.meters_per_pixel,
        )

    def to_pixel(self, world: Vec2) -> Vec2:
        if not all(isfinite(v) for v in (world.x, world.y)):
            raise ValueError("world position must be finite")
        return Vec2(
            self.origin_px.x + world.x / self.meters_per_pixel,
            self.origin_px.y - world.y / self.meters_per_pixel,
        )

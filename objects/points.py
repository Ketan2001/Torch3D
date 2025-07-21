from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scene.scene import Scene


_ZERO_THRESHOLD = 1e-8 # w < thresold is considered to be ~0, making it a point at infinity


class BasePoint:
    """
    Base class for a Point in N dimensional Euclidean space, represented in homogenous coordinates (N dimensional Projective Space)
    """
    def __init__(
        self,
    ):
        pass


class Point3D(BasePoint):
    def __init__(
        self,
        scene: 'Scene',
        idx: int
    ):
        self.scene = scene
        self.idx = idx

    @property
    def homogenous(self):
        return self.scene.points[self.idx].view(-1, 1) # coords are always column vectors
    
    @property
    def inhomogenous(self):
        coords = self.homogenous
        w = coords[-1, 0].item()
        if abs(w) < _ZERO_THRESHOLD:
            # handle edge case when point is at infinity
            raise NotImplementedError("Processing of points at infinity is currently not handled")

        return coords[:-1] / w



class Point2D(BasePoint):
    def __init__(self):
        pass
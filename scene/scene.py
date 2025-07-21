import torch
from objects.points import Point3D
from typing import Literal
from utils.helper import get_4d_transform


class Scene:
    def __init__(self, renderer: Literal["mesh", "plt"] = "plt"):
        self.points = torch.empty((0, 4, 1))
        self.edges = []

    def add_point(self, data: torch.Tensor | list):
        if isinstance(data, list):
            data = torch.tensor(data)
        
        data = data.view(-1)
        if data.shape[0] == 3:
            # append 1. to homogenize
            data = torch.cat([data, torch.ones(1)], axis=-1)
        elif data.shape[0] == 4:
            # assuming input is already homogenized
            pass
        else:
            raise ValueError(f"Points need to have at least 3 coordinates, found {data}, having only {data.shape[0]} coordinates")
        
        self.points = torch.cat(
            [self.points, data[None, : , None]], dim=0
        )

        # get the idx of this point
        idx = self.points.shape[0] - 1

        # create a Point3D object that points to this scene and idx
        point = Point3D(idx = idx, scene = self)

        return point

    def add_edge(self, edge: tuple[int]):
        self.edges.append(edge)
    
    def transform_points(self, R: torch.Tensor, t: torch.Tensor):
        tform = get_4d_transform(R, t)
        self.points = tform @ self.points
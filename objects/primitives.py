import torch

"""
All of these functions returns point coordinates and 
edges for common 3D shapes
"""


def get_cube() -> tuple[list[list[float]], list[tuple[int]]]:
    points = [
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 1.],
        [0., 0., 1.],
    ]

    edges = []

    for i in range(len(points)):
        point = points[i]
        for j in range(i + 1, len(points), 1):
            point2 = points[j]
            if torch.abs(
                torch.tensor(point2) - torch.tensor(point)
            ).sum() == 1:
                edges.append((i, j))
    
    return points, edges


def get_double_pyramid() -> tuple[list[list[float]], list[tuple[int]]]:
    points = [
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [1., 1., 0.],
    ]

    edges = []

    for i in range(len(points)):
        point = points[i]
        for j in range(i + 1, len(points), 1):
            point2 = points[j]
            if torch.abs(
                torch.tensor(point2) - torch.tensor(point)
            ).sum() == 1:
                edges.append((i, j))
    
    return points, edges
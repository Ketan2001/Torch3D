"""
A Camera object consists of its position and orientation with respect to the word frame
It also contains the camera intrinsics, allowing for methods that allow coordinate transforms
from world to camera frame
"""

import torch
from objects.points import Point3D, Point2D
from typing import Literal


class Camera:
    def __init__(
        self, 
        R: torch.Tensor, 
        t: torch.Tensor, 
        f: float = 0.1,
        c_x: float = 256,
        c_y: float = 256,
        m: float = 3000,
        device: torch.device = torch.device("cpu"),
        type: Literal["persp", "ortho"] = "persp"
    ):
        """
        args:
        R: 3D rotation matrix of shape (3,3)
        t: 3D translation vector of shaoe (3, 1)
        f: focal length (magnitude only sign is handled internally)
        c_x: 
        c_y: 
        """
        assert f > 0.

        self.f = f
        self.R = R.to(device) # (3, 3)
        self.t = t.to(device) # (3, 1)
        self.K = torch.tensor([
            [f * m, 0., c_x],
            [0., -f * m, c_y],
            [0., 0., 1.]
        ]).to(device)

        self.M = torch.cat([self.R, self.t], dim=-1)

        if type == 'persp':
            self.P = self.K @ self.M # (3, 4)
        
        else:
            raise ValueError(f"Camera type {type} not recognized")
    
    @property
    def proj_matrix(self):
        return (self.K @ self.M).unsqueeze(0)
    
    @property
    def camera_transform(self):
        return self.M
    
    @property
    def intrinsic_matrix(self):
        return self.K
    
    def project(self, points: torch.Tensor):
        """
        args:
            points: collection of points in 3D projective space, (N, 4, 1)

        returns:
            coordinates in camera frame (N, 3, 1)
        """
        projected_points = self.P @ points # (N, 3, 1)
        z_coords = projected_points[:, 2] # (N, 1)

        # points in front of the camera
        valid_points = torch.where(z_coords > self.f)[0]
        
        projected_points = projected_points / projected_points[:, 2:]

        return projected_points, z_coords, valid_points
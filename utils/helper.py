import torch
from typing import Literal
import math


def get_skey_symm_matrix(u):
    assert u.shape == (3, 1)
    u_1 = u[0, 0]
    u_2 = u[1, 0]
    u_3 = u[2, 0]
    
    u_hat = torch.tensor(
        [
            [ 0.0,  -u_3,   u_2],
            [ u_3,   0.0,  -u_1],
            [-u_2,   u_1,   0.0]
        ]
    )

    return u_hat


def cross_product(u, v):
    assert u.shape == (3, 1) and v.shape == (3, 1)
    
    u_hat = get_skey_symm_matrix(u)
    return u_hat @ v


def cross_product_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    batch cross prodict of 2d vectors
    params:
        u: torch.Tensor of shape (N, 2) or (1, 2)
        v: torch.Tensor of shape (N, 2) or (1, 2)
    returns:
        torch.Tensor of shape (N, 2)
    """
    return ((u * torch.flip(v, dims=(-1,))) @ torch.tensor([[1.], [-1.]], device=v.device)).squeeze(-1)


def cross_product_3d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    batch cross prodict of 2d vectors
    params:
        u: torch.Tensor of shape (N, 3) or (1, 3)
        v: torch.Tensor of shape (N, 3) or (1, 3)
    returns:
        torch.Tensor of shape (N, 3)
    """
    x = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    y =  u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    z =  u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    return torch.cat([x, y, z], dim=-1)


def get_rotation_mat(w, theta, mode: Literal["degree", "rad"] = "rad"):
    """
    Implements the Rodrigues' formula
    exp[theta * w_hat] = I + w_hat * sin(theta) + (w_hat ** 2) * (1 - cos(theta))
                       = I + w_hat * sin(theta) + (w w^T - I) * (1 - cos(theta))
    """
    assert w.shape == (3, 1)
    if mode == "degree":
        theta = theta * (torch.pi / 180.)
    
    # Normalize the axis
    w_normalized = w / torch.norm(w)
    w_hat = get_skey_symm_matrix(w_normalized)

    return torch.eye(3) + w_hat * math.sin(theta) + (w_hat @ w_hat) * (1. - math.cos(theta)) # + w @ w.T - torch.identity(3) * (1. - torch.cos(theta))


def read_3d_vector(x: list | torch.Tensor, device: torch.device = torch.device("cpu")):
    if isinstance(x, list):
        x = torch.tensor(x, device=device)
    
    x = x.view(-1)
    assert x.shape[0] == 3, f'Expecting 3D vector to have 3 coordinates, found {x}, having {x.shape[0]} coordintes'

    return x.reshape(-1, 1)


def get_4d_transform(R: torch.tensor, t: torch.tensor):
    R = R.view(3, 3)
    t = t.view(3, 1)

    tform = torch.cat([
        R, t
    ], dim=-1) # (3, 4)

    tform = torch.cat([
        tform, torch.tensor([[0., 0., 0., 1]])
    ], dim=0)

    return tform[None, ...]


def get_look_at_transform(
    position: list | torch.Tensor, 
    target: list | torch.Tensor, 
    up: list | torch.Tensor
):
    """
    args:
        position: 3D position of the camera in world space
        target: 3d position for camera to look at
        up: camera up direction, defaults to world up (0., 1., 0.)
    
    returns:
        R_cw, t_cw that transform from world to camera coordinates
    """
    position = read_3d_vector(position)
    target = read_3d_vector(target)
    up = read_3d_vector(up)

    up /= torch.norm(up)

    # camera looks down Z
    Z = target - position
    Z /= torch.norm(Z)

    # check if Z aligns with up direction
    dot = Z.T @ up
    if torch.abs(dot) > 0.99:
        raise RuntimeError("Ambiguous Camera Up, look at direction and up are aligned, not handled!")

    X = cross_product(torch.tensor([0., 1., 0.]).unsqueeze(-1), Z)
    X /= torch.norm(X)

    # now we can get the camera up
    Y = cross_product(Z, X)

    R_cw = torch.cat([X, Y, Z], dim=1).T

    T_cw = -1 * R_cw @ position

    return R_cw, T_cw
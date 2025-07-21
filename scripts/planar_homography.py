from scene.scene import Scene
from scene.renderer import MatplotlibRenderer
from objects.camera import Camera
from utils.helper import get_look_at_transform
import torch


scene = Scene()
renderer =  MatplotlibRenderer()

# add a plane to the scene
plane = [
    [1., 1., 0.],
    [-1., 1., 0.],
    [1., -1., 0.],
    [-1., -1., 0.]
]

for point in plane:
    scene.add_point(
        point
    )

scene.add_edge((0, 1))
scene.add_edge((0, 2))
scene.add_edge((1, 3))
scene.add_edge((2, 3))


R, t = get_look_at_transform(
    position=[4., 3., 5],
    target=[0., 0., 0.],
    up=[0., 1., 0]
)
cam1 = Camera(
    R=R,
    t=t,
    f=0.1,
)

R, t = get_look_at_transform(
    position=[2., -3., 7],
    target=[0., 0., 0.],
    up=[0., 1., 0]
)
cam2 = Camera(
    R=R,
    t=t,
    f=0.1,
)


H = torch.randn((4, 4))
H = torch.randn((4, 4))
H_inv = torch.linalg.inv(H)

cam1.P = cam1.P @ H
cam2.P = cam2.P @ H

scene.points = H_inv[None, ...] @ scene.points

proj_points_1, z_cam_1, mask_1 = cam1.project(scene.points)
proj_points_2, z_cam_2, mask_2 = cam2.project(scene.points)

proj_points_1_est = (z_cam_2 / z_cam_1)[..., None] \
    * cam1.proj_matrix[:, :, [0, 1, 3]] \
    @ torch.linalg.inv(cam2.proj_matrix[:, :, [0, 1, 3]]) @ proj_points_2 # (4, 3, 1)

renderer.render(proj_points_1[:, :2, 0], 'cam1.png', scene.edges, mask=mask_1)
renderer.render(proj_points_1_est[:, :2, 0], 'cam1_est.png', scene.edges, mask=mask_1)
renderer.render(proj_points_2[:, :2, 0], 'cam2.png', scene.edges, mask=mask_2)
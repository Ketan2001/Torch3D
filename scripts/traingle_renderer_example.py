from scene.scene import Scene
from scene.renderer import TriangleRenderer
from objects.camera import Camera
import torch
import math
import time
import cv2
import numpy as np
from utils.helper import get_look_at_transform, get_rotation_mat
from utils.io import read_obj


scene = Scene()
renderer = TriangleRenderer()

vertices, faces = read_obj('./assets/sphere.obj')

faces = torch.tensor(faces, dtype=torch.int32)

for point in vertices:
    scene.add_point(point)


colors = torch.rand((len(scene.points), 3))


for i in range(100):
    R, t = get_look_at_transform(
        position=[5 * math.sin(i / 5), 0, 5 * math.cos(i / 5)],
        target=[0., 0., 0],
        up=[0., 1., 0]
    )
    cam = Camera(R, t, c_x=1980 / 2, c_y=1080/2)

    verts_proj, z_coords, valid_verts = cam.project(scene.points)

    # get face indices that have all three vertices in front of the camera
    valid_face_indices = torch.where(
        torch.all(torch.isin(faces, valid_verts), dim=-1)
    )[0]

    img = renderer.render(
        vertex_projections=verts_proj[:, :2, 0],
        z_coords=z_coords,
        faces=faces[valid_face_indices],
        colors=colors,
        resolution=(1080, 1920) # (H, W)
    )

    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)

    cv2.imshow('Triangle Renderer', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)

cv2.destroyAllWindows()
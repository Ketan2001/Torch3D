from scene.scene import Scene
from scene.renderer import TriangleRenderer
from objects.camera import Camera
import torch
from utils.helper import get_look_at_transform
from utils.io import read_obj, save_image


scene = Scene(ambient_strength=0.1)

scene.add_light(direction=[-1., -1., -1.])
scene.add_light(direction=[0., -1., 0.])

renderer = TriangleRenderer(scene)

vertices, faces = read_obj('./assets/stanford_bunny.obj')

faces = torch.tensor(faces, dtype=torch.int32)

for point in vertices:
    scene.add_point(point)

colors = torch.ones((len(scene.points), 3))

R, t = get_look_at_transform(
    position=[0., 0., 3.],
    target=[0., 0., 0],
    up=[0., 1., 0]
)

canvas = (1080, 1920)

cam = Camera(R, t, c_x=canvas[1] / 2, c_y=canvas[0] / 2)

verts_proj, z_coords, valid_verts = cam.project(scene.points)

# get face indices that have all three vertices in front of the camera
valid_face_indices = torch.where(
    torch.all(torch.isin(faces, valid_verts), dim=-1)
)[0]

img = renderer.render(
    verts=scene.points[:, :3, :],
    vertex_projections=verts_proj[:, :2, 0],
    z_coords=z_coords,
    faces=faces[valid_face_indices],
    colors=colors,
    resolution=canvas # (H, W)
)

save_image(img, 'rabbit.png')
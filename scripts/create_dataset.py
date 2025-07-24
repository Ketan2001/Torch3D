from scene.scene import Scene
from scene.renderer import TriangleRenderer
from objects.camera import Camera
import torch
from utils.helper import get_look_at_transform
from utils.io import read_obj, save_image, read_image
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import random


@dataclass
class DatasetConifg:
    out_dir: Path

    num_views: int = 500

    # camera intrinsics
    f: float = 0.4
    m: float = 1000.
    c_x: float = 128.
    c_y: float = 128.
    canvas_size: tuple[int] = field(default_factory=lambda: (256, 256))

    # scene
    obj_file_path: str = "assets/stanford_bunny.obj"
    lights: list[list[int]] = field(default_factory=lambda: [
        [-1., -1., -1., .5],
        [ 0., -1.,  0., .2],
        [ 1.,  1.,  1., .5],
    ])
    ambient_strenght: float = 0.2


def setup_scene(config: DatasetConifg, device: torch.device):
    scene = Scene(ambient_strength=config.ambient_strenght, device=device)

    for item in config.lights:
        scene.add_light(direction=item[:3], strength=item[-1])

    vertices, faces = read_obj(config.obj_file_path)

    faces = torch.tensor(faces, dtype=torch.int32, device=device)
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)

    scene.points = torch.cat(
        [vertices, torch.ones((len(vertices), 1), dtype=torch.float32, device=device)], dim=-1
    ).unsqueeze(-1)
    
    # center the object to world origin
    centre = (
        scene.points[:, :3, 0].min(dim=0,keepdim=True)[0] + \
            scene.points[:, :3, 0].max(dim=0,keepdim=True)[0]
    ) * .5

    scene.points[:, :3, 0] -= centre

    return scene, faces


def main(config: DatasetConifg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n=> Using Device : {device}\n")
    
    scene, faces = setup_scene(config, device=device)

    renderer = TriangleRenderer(scene)

    colors = torch.ones((len(scene.points), 3), device=device)

    for view_idx in tqdm(range(config.num_views), desc="Generating Dataset"):
        # randomly pick a camera location appropritely positioned around the object
        # fist in polar coordinates pick an r and a theta
        r = 6.5 + np.random.rand() * 2
        theta = np.random.rand() * 2 * np.pi

        R, t = get_look_at_transform(
            position=torch.tensor([r * np.sin(theta), np.random.rand() * 2, r * np.cos(theta)], dtype=torch.float32), 
            target=[0., 0., 0.],
            up=[0., 1., 0]
        )

        cam = Camera(
            R, 
            t, 
            f=config.f, 
            m=config.m, 
            c_x=config.c_x, 
            c_y=config.c_y,
            device=device
        )

        verts_proj, z_coords, valid_verts = cam.project(scene.points)

        # get face indices that have all three vertices in front of the camera
        valid_face_indices = torch.where(
            torch.all(torch.isin(faces, valid_verts), dim=-1)
        )[0]

        if len(valid_face_indices) != len(faces):
            raise RuntimeError("some points are behind the camera, this is unexpected for this setting, \
                position the cameras at an appropriate distance")

        img, ids = renderer.render(
            verts=scene.points[:, :3, :],
            vertex_projections=verts_proj[:, :2, 0],
            z_coords=z_coords,
            faces=faces,
            colors=colors,
            resolution=config.canvas_size, # (H, W)
        )

        config.out_dir.mkdir(exist_ok=True)

        # save the image
        save_image(img, config.out_dir / f'{view_idx}.png')

        # pixel locations corresponding to objects
        valid_pixels = torch.vstack(torch.where(ids[:, :, 0] > -1)).transpose(0, 1).detach().cpu().numpy()

        # barycentric coordinates for these pixels
        valid_descriptors = ids[torch.where(ids[:, :, 0] > -1)][:, 1:].detach().cpu().numpy()
        tri_indices = ids[torch.where(ids[:, :, 0] > -1)][:, 0]

        id_dict = defaultdict(list)
        for idx, tri_idx in enumerate(tri_indices):
            id_dict[int(tri_idx.item())].append(
                (
                    valid_pixels[idx].astype(np.uint32).tolist(), 
                    np.round(valid_descriptors[idx], decimals=3).tolist()
                )
            )
        
        # save pixel wise "labels" for GT correspondences
        with open(config.out_dir / f'{view_idx}_pixel_labels.json', "w") as f:
            json.dump(id_dict, f, indent=4)

        # save camera extrinsics
        np.savetxt(
            config.out_dir / f'{view_idx}_camera_mat.txt',
            cam.camera_transform.detach().cpu().numpy()
        )


def vis_correspondences(dataset_dir: str | Path):
    """
    Sample two random views from a dataset
    get all correspondences
    plot them side by side
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
    
    image_paths = list(dataset_dir.glob("*.png"))
    view_indices = [int(p.stem) for p in image_paths]

    view_1, view_2 = random.choices(view_indices, k=2)

    img_1 = read_image(img_path=dataset_dir / f'{view_1}.png')
    img_2 = read_image(img_path=dataset_dir / f'{view_2}.png')

    with open(dataset_dir / f"{view_1}_pixel_labels.json", "r") as f:
        labels_view_1 = json.load(f)
    
    with open(dataset_dir / f"{view_2}_pixel_labels.json", "r") as f:
        labels_view_2 = json.load(f)

    keys_label_1 = list(labels_view_1.keys())
    keys_label_2 = list(labels_view_2.keys())

    common_keys = set(keys_label_1).intersection(set(keys_label_2))

    correspondences = []

    for key in common_keys:
        locs_view_1 = labels_view_1[key]
        locs_view_2 = labels_view_2[key]

        # cases where a triangle only covered a single pixel in both views
        if len(locs_view_1) == 1 and len(locs_view_2) == 1:
            correspondences.append(
                [locs_view_1[0][0], locs_view_2[0][0]]
            )
    
    # sample any 10 and put red dots
    for idx, (x, x_dash) in enumerate(correspondences):
        img_1[x[0], x[1]] = [255, 0, 0]
        img_2[x_dash[0], x_dash[1]] = [255, 0, 0]
    
    save_image(img_1, 'view_1.png')
    save_image(img_2, 'view_2.png')


if __name__ == '__main__':
    # main(config=DatasetConifg(out_dir=Path("bunny_dataset")))
    vis_correspondences("bunny_dataset")
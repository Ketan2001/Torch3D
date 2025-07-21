import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MatplotlibRenderer:
    def __init__(
        self,
    ):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    def render(
        self,
        pixel_coords: torch.Tensor, # (N, 2)
        out_path: Path | str,
        edges: list = [], # list of tuples of edges
        mask = None,
    ):
        pixel_coords_np = pixel_coords.cpu().numpy()

        if len(mask) == 0:
            print("No points in front of the camera")
            return

        plt.figure()
        plt.axis('equal')
        plt.scatter(pixel_coords_np[mask, ...][:, 0], pixel_coords_np[mask, ...][:, 1])

        if len(edges):
            for color_idx, (idx1, idx2) in enumerate(edges):
                if idx1 in mask and idx2 in mask:
                    # [x1, x2], [y1, y2]
                    plt.plot(
                        [pixel_coords_np[idx1, 0], pixel_coords_np[idx2, 0]],
                        [pixel_coords_np[idx1, 1], pixel_coords_np[idx2, 1]],
                        color=self.colors[color_idx % len(self.colors)]
                    )
        plt.savefig(out_path)
        plt.close()


class TriangleRenderer:
    def __init__(
        self
    ):
        pass

    def render(
        self, 
        vertex_projections: torch.Tensor, # (num_verts, 2)
        z_coords: torch.Tensor, # (num_verts, 1)
        faces: torch.Tensor, # (num_traingles, 3)
        colors: torch.Tensor, # (num_verts, 3),
        resolution: tuple[int] = (512, 512)
    ):
        # this flip accounts for the fact that images are indexed [rows, columns]
        # where rows -> down, columns -> right in image space
        # the vertex_projections have x-coords -> right, y-coord -> down
        vertex_projections = torch.flip(vertex_projections, [1])
        image = torch.zeros((resolution[0], resolution[1], 3)) # RGB image

        depth_buffer = torch.full(resolution, fill_value=torch.finfo(torch.float32).max, dtype=torch.float32)

        # for each traingle, get a bounding box
        face_verts = vertex_projections[faces] # (num_traingles, 3, 2)
        face_colors = colors[faces] # (num_triangles, 3, 3)
        face_depths = z_coords[faces] # (num_traingles, 3, 1)

        verts_x = face_verts[:, :, 0]
        verts_y = face_verts[:, :, 1]

        min_x = torch.floor(verts_x.min(axis=1)[0]).clamp(0, resolution[0]).to(int)
        max_x = torch.ceil(verts_x.max(axis=1)[0]).clamp(0, resolution[0]).to(int)

        min_y = torch.floor(verts_y.min(axis=1)[0]).clamp(0, resolution[1]).to(int)
        max_y = torch.ceil(verts_y.max(axis=1)[0]).clamp(0, resolution[1]).to(int)

        # this narrows down the pixels we need to test for
        # we process traingles sequentially, keeping a depth value stored for each pixel if it is coloured
        # if we get a tiangle with lower depth for a pixel, we overwrite the color
        for tri_idx, tri in enumerate(face_verts):
            v0, v1, v2 = tri[0:1], tri[1:2], tri[2:]

            x_coords = torch.arange(
                min_x[tri_idx], max_x[tri_idx], 1
            , dtype=torch.int32)
            y_coords = torch.arange(
                min_y[tri_idx], max_y[tri_idx], 1
            , dtype=torch.int32)

            grid_x = x_coords.reshape(-1, 1).expand(-1, len(y_coords))
            grid_y = y_coords.reshape(1, -1).expand(len(x_coords), -1)

            points_test = torch.cat(
                [grid_x[..., None], grid_y[..., None]], 
                dim=-1
            ).reshape(-1, 2)

            e_01 = TriangleRenderer.edge_func(v_s=v0, v_e=v1, p=points_test)
            e_12 = TriangleRenderer.edge_func(v_s=v1, v_e=v2, p=points_test)
            e_20 = TriangleRenderer.edge_func(v_s=v2, v_e=v0, p=points_test)

            valid_indices = torch.where(
                (e_01 > 0) & (e_12 > 0) & (e_20 > 0)
            )[0]

            if not len(valid_indices):
                continue

            valid_pixels = points_test[valid_indices]

            # for the valid pixels compute barycentric coords
            bary = torch.stack(
                [e_12, e_20, e_01], dim=-1
            )

            bary /= bary.sum(-1, keepdim=True)

            bary_valid = bary[valid_indices]

            # compute depths

            # incorrect way (values don't vary linearly beacause of the perspective divide)
            # depths = (bary_valid @ face_depths[tri_idx]).squeeze(-1)

            inverse_depths = (bary_valid @ (1. / face_depths[tri_idx]))
            depths = (1. / inverse_depths).squeeze(-1)

            _ind = torch.where(
                depths - depth_buffer[valid_pixels[:, 0], valid_pixels[:, 1]]
            )[0] 

            depth_buffer[valid_pixels[_ind, 0], valid_pixels[_ind, 1]] = depths[_ind]

            valid_depth_filtered = valid_indices[_ind]

            bary_valid = bary[valid_depth_filtered]
            valid_pixels = points_test[valid_depth_filtered]

            # perpsective correct interpolation of color
            colors_by_z = face_colors[tri_idx] / face_depths[tri_idx]
            interp_colors = (bary_valid @ colors_by_z[None, ...]) / inverse_depths
            image[valid_pixels[:, 0], valid_pixels[:, 1]] = interp_colors
        
        return image

    @staticmethod
    def edge_func(v_s, v_e, p):
        """
        v_s: Starting point of edge tensor of shape (1, 2)
        v_e: End point of edge tensor of shape (1, 2)
        p: test_points, tensor of shape (N, 2)

        returns:
            cross product (p - v_s) x (v_e - v_s)
        """
        return (((p - v_s) * torch.flip((v_e - v_s), dims=(-1,))) @ torch.tensor([[1.], [-1.]])).squeeze(-1)
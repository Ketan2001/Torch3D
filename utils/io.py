from pathlib import Path
import cv2
import numpy as np
import torch


def read_obj(path: str | Path):
    f = open(path, 'r')

    vertices = []
    faces = []

    for line in f:
        line = line.replace("\n", "")
        if line.startswith("v "):
            # this line has a vertex
            parts = line.split(' ')
            vertices.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )
        if line.startswith("f "):
            # this line has a face
            parts = line.split(' ')
            vert_indices = parts[1:]
            
            if len(vert_indices) != 3:
                raise NotImplementedError("Currently only traingle faces are supported")

            face = []

            for vert_ind in vert_indices:
                # -1 to account for the fact that faces are 1 indexed in obj files
                face.append(int(vert_ind.split('//')[0]) - 1)
            
            faces.append(face)
    
    f.close()

    return vertices, faces


def read_image(img_path: str | Path) -> np.ndarray:
    """
    Returns image as RGB np.ndarray of shape (H, W, C), in range (0, 1.)
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(img: torch.Tensor | np.ndarray, output_path: str | Path):
    """
    args:
        img: float image tensor or numpy array assumed to have valid range [0., 1.]
        output_path: path to save image
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    if img.dtype == np.float32:
        img = img.clip(0., 1.) * 255.
    
    if img.ndim == 3:
        # image is RGB or grayscale with channel dimension already added
        if img.shape[0] == 3:
            # we have C, H, W image
            img = np.transpose(img, (1, 2, 0))
    else:
        raise ValueError(f"Image array has ndim {img.ndim}, expected 3")

    img = img.astype(np.uint8)

    assert img.min() >= 0 and img.max() <= 255, f"Image tensor must have values in \
        valid range, clip before passing to this function"

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    vertices, faces = read_obj('./assets/sphere.obj')
    
    print(vertices)
    print(faces)
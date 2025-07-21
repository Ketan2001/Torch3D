def read_obj(path):
    f = open(path, 'r')

    vertices = []
    faces = []

    for line in f:
        if line.startswith("v "):
            # this line has a vertex
            parts = line.split(' ')
            vertices.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )
        if line.startswith("f "):
            # this line has a face
            parts = line.split(' ')
            vert_indices = parts[1:-1]

            if len(vert_indices) != 3:
                raise NotImplementedError("Currently only traingle faces are supported")

            face = []

            for vert_ind in vert_indices:
                # -1 to account for the fact that faces are 1 indexed in obj files
                face.append(int(vert_ind.split('/')[0]) - 1)
            
            faces.append(face)
    
    f.close()

    return vertices, faces


if __name__ == '__main__':
    vertices, faces = read_obj('./assets/sphere.obj')
    
    print(vertices)
    print(faces)
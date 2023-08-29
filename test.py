import torch
import numpy as np
from RayTracer import RayTracer
import trimesh

def write_ply(v, path):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]

    with open(path, 'w') as meshfile:
        meshfile.write(f'{header}{"".join(str_v)}')

vertices = np.array([
    -0.57735,  -0.57735,  0.57735,
    0.934172,  0.356822,  0,
    0.934172,  -0.356822,  0,
    -0.934172,  0.356822,  0,
    -0.934172,  -0.356822,  0,
    0,  0.934172,  0.356822,
    0,  0.934172,  -0.356822,
    0.356822,  0,  -0.934172,
    -0.356822,  0,  -0.934172,
    0,  -0.934172,  -0.356822,
    0,  -0.934172,  0.356822,
    0.356822,  0,  0.934172,
    -0.356822,  0,  0.934172,
    0.57735,  0.57735,  -0.57735,
    0.57735,  0.57735,  0.57735,
    -0.57735,  0.57735,  -0.57735,
    -0.57735,  0.57735,  0.57735,
    0.57735,  -0.57735,  -0.57735,
    0.57735,  -0.57735,  0.57735,
    -0.57735,  -0.57735,  -0.57735,
    ]).reshape((-1,3), order="C")

faces = np.array([
    19, 3, 2,
    12, 19, 2,
    15, 12, 2,
    8, 14, 2,
    18, 8, 2,
    3, 18, 2,
    20, 5, 4,
    9, 20, 4,
    16, 9, 4,
    13, 17, 4,
    1, 13, 4,
    5, 1, 4,
    7, 16, 4,
    6, 7, 4,
    17, 6, 4,
    6, 15, 2,
    7, 6, 2,
    14, 7, 2,
    10, 18, 3,
    11, 10, 3,
    19, 11, 3,
    11, 1, 5,
    10, 11, 5,
    20, 10, 5,
    20, 9, 8,
    10, 20, 8,
    18, 10, 8,
    9, 16, 7,
    8, 9, 7,
    14, 8, 7,
    12, 15, 6,
    13, 12, 6,
    17, 13, 6,
    13, 1, 11,
    12, 13, 11,
    19, 12, 11,
    ]).reshape((-1, 3), order="C")-1

# vertices = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
# faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)

# vertices = np.concatenate([vertices, vertices-2, vertices+2], axis=0)
# faces = np.concatenate([faces, faces + 4, faces+8], axis=0)

if __name__ == '__main__':
    raytracer = RayTracer(vertices, faces)
    # xyz = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).float()
    # dir = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()

    face_verts = vertices[faces]
    dir = face_verts.mean(axis=1)
    xyz = np.ones_like(dir)
    xyz = torch.from_numpy(xyz)
    dir = torch.from_numpy(dir)

    intersection, normal, depth, face_idx = raytracer.trace(xyz, dir)
    print(intersection.shape)
    print(normal.device)
    print(depth)
    print(face_idx)
    mask = depth < 9.
    intersection, normal, depth, face_idx = intersection[mask], normal[mask], depth[mask], face_idx[mask]
    print(intersection, normal, depth)
    print(face_idx.cpu().numpy())
    faces_inter = faces[face_idx.cpu().numpy()]
    trimesh.Trimesh(vertices=vertices, faces=faces).export('./test_data/object.obj')
    trimesh.Trimesh(vertices=vertices, faces=faces_inter).export('./test_data/intersected_faces.obj')
    write_ply(intersection.cpu().numpy(), './test_data/intersections.ply')

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=xyz.numpy(), ray_directions=dir.numpy(), multiple_hits=False)
    face_idx_pyrender = []
    intersection = intersection.cpu().numpy()
    for i in range(locations.shape[0]):
        face_idx_pyrender.append(index_tri[np.argmin(((locations - intersection[i]) ** 2).sum(axis=-1))])
    face_idx_pyrender = np.stack(face_idx_pyrender, axis=0)
    print(face_idx_pyrender)
    write_ply(locations, './test_data/intersections_pyrender.ply')
    trimesh.Trimesh(vertices=vertices, faces=faces[index_tri]).export('./test_data/intersected_faces_pyrender.obj')

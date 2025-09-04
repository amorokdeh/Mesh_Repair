import numpy as np
import pyvista as pv
from numba import njit, prange

class Vertex:
    def __init__(self, coords, index):
        self.coords = coords  # np.array([x,y,z])
        self.index = index
        self.valence = 0
        self.normal = None
        self.triangle_indices = []

class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1  # vertex indices
        self.v2 = v2
        self.triangles = []  # max two triangle indices; -1 if border

class Triangle:
    def __init__(self, vertex_indices, index):
        self.vertex_indices = vertex_indices  # 3 vertex indices
        self.edge_indices = [-1, -1, -1]  # to be assigned later
        self.index = index
        self.normal = None

    def recompute_normal(self, vertices):
        v0 = vertices[self.vertex_indices[0]].coords
        v1 = vertices[self.vertex_indices[1]].coords
        v2 = vertices[self.vertex_indices[2]].coords

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        self.normal = normal / norm if norm != 0 else np.array([0, 0, 0])

# ------------------------------
# Numba-optimized function for triangle normals
@njit(parallel=True)
def compute_triangle_normals(vertices_coords, triangles_idx):
    n_tri = triangles_idx.shape[0]
    normals = np.zeros((n_tri, 3))
    for i in prange(n_tri):
        v0 = vertices_coords[triangles_idx[i, 0]]
        v1 = vertices_coords[triangles_idx[i, 1]]
        v2 = vertices_coords[triangles_idx[i, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals[i] = n
    return normals

# ------------------------------
def build_mesh_from_stl(file_path, progress_callback=None):
    mesh = pv.read(file_path)
    points = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # triangular faces only

    n_points = len(points)
    n_faces = len(faces)
    total_steps = (
        n_points +          # vertex creation
        n_faces +           # triangle creation
        n_faces +           # edge generation
        n_faces * 3 +       # normal computation
        n_faces * 3         # vertex updates
    )
    step = 0

    # --- Build vertices ---
    vertices = [Vertex(coords=points[i], index=i) for i in range(n_points)]
    step += n_points
    if progress_callback:
        progress_callback("Building vertices...")

    # --- Build triangles ---
    triangles = [Triangle(vertex_indices=faces[i].tolist(), index=i) for i in range(n_faces)]
    step += n_faces
    if progress_callback:
        progress_callback("Building triangles...")

    # --- Build unique edges ---
    edge_dict = {}
    edges = []
    for tri in triangles:
        vids = tri.vertex_indices
        edge_vertices = [(vids[1], vids[2]), (vids[2], vids[0]), (vids[0], vids[1])]
        for i, (v1, v2) in enumerate(edge_vertices):
            key = tuple(sorted((v1, v2)))
            if key in edge_dict:
                ei = edge_dict[key]
                edges[ei].triangles.append(tri.index)
                tri.edge_indices[i] = ei
            else:
                ei = len(edges)
                e = Edge(v1=key[0], v2=key[1])
                e.triangles.append(tri.index)
                edges.append(e)
                edge_dict[key] = ei
                tri.edge_indices[i] = ei
        step += 1
        if step % 100 == 0 and progress_callback:
            percent = int((step / total_steps) * 100)
            progress_callback(f"{percent}")
    if progress_callback:
        progress_callback("Building edges...")

    # --- Update vertex data ---
    for tri in triangles:
        for v_idx in tri.vertex_indices:
            vertices[v_idx].valence += 1
            vertices[v_idx].triangle_indices.append(tri.index)
        step += 1
        if step % 100 == 0 and progress_callback:
            percent = int((step / total_steps) * 100)
            progress_callback(f"{percent}")
    if progress_callback:
        progress_callback("Assigning triangle refs...")

    # --- Compute triangle normals using Numba ---
    verts_coords = np.array([v.coords for v in vertices])
    tris_idx = np.array([tri.vertex_indices for tri in triangles])
    normals = compute_triangle_normals(verts_coords, tris_idx)
    for i, tri in enumerate(triangles):
        tri.normal = normals[i]
        step += 1
        if step % 200 == 0 and progress_callback:
            percent = int((step / total_steps) * 100)
            progress_callback(f"{percent}")

    if progress_callback:
        progress_callback("100")

    return vertices, edges, triangles
import numpy as np
import pyvista as pv

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

# mesh_data_structure.py
def build_mesh_from_stl(file_path, progress_callback=None):
    mesh = pv.read(file_path)
    points = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # assuming triangular mesh

    total_steps = (
    len(points) +          # vertex creation
    len(faces) +           # triangle creation
    len(faces) +           # edge generation (loop once per triangle)
    len(faces) * 3 +       # normal computation (once per triangle)
    len(faces) * 3         # vertex updates (each triangle has 3 vertices)
    )
    step = 0

    # Build vertices list
    vertices = []
    for i in range(len(points)):
        vertices.append(Vertex(coords=points[i], index=i))
        step += 1
        if i % 100 == 0 and progress_callback:
            percent = min(100, int((step / total_steps) * 100))
            progress_callback(f"Building vertices... {percent}%")

    # Build triangles list
    triangles = []
    for i, face in enumerate(faces):
        triangles.append(Triangle(vertex_indices=face.tolist(), index=i))
        step += 1
        if i % 100 == 0 and progress_callback:
            percent = int((step / total_steps) * 100)
            progress_callback(f"Building triangles... {percent}%")

    # Build Unique Edges and Link to Triangles
    edge_dict = {}  # Helps avoid duplicate edge creation
    edges = []  # List of unique Edge instances

    for tri in triangles:
        vids = tri.vertex_indices
        edge_vertices = [
            (vids[1], vids[2]),
            (vids[2], vids[0]),
            (vids[0], vids[1]),
        ]
        for i, (v_start, v_end) in enumerate(edge_vertices):
            key = tuple(sorted((v_start, v_end))) # Sort to ensure uniqueness (undirected edge)
            if key in edge_dict:
                # Edge already exists, reuse it
                edge_index = edge_dict[key]
                edges[edge_index].triangles.append(tri.index)
                tri.edge_indices[i] = edge_index
            else:
                # Create new edge and assign it
                edge_index = len(edges)
                edge = Edge(v1=key[0], v2=key[1])
                edge.triangles.append(tri.index)
                edges.append(edge)
                edge_dict[key] = edge_index
                tri.edge_indices[i] = edge_index
        step += 1
    if progress_callback:
        progress_callback("Building edges...")

    # Update Vertex Data with Triangle Relationships
    for tri in triangles:
        for v_idx in tri.vertex_indices:
            vertices[v_idx].valence += 1  # Increase how many triangles touch this vertex
            vertices[v_idx].triangle_indices.append(tri.index) # Store triangle ID
        step += 1
    if progress_callback:
        progress_callback("Assigning triangle refs...")

    # Compute Triangle Surface Normals
    for tri in triangles:
        p0 = vertices[tri.vertex_indices[0]].coords
        p1 = vertices[tri.vertex_indices[1]].coords
        p2 = vertices[tri.vertex_indices[2]].coords
        edge1 = p1 - p0
        edge2 = p2 - p0
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        tri.normal = n / norm if norm > 0 else np.array([0,0,0])
        step += 1
        if step % 200 == 0 and progress_callback:
            percent = int((step / total_steps) * 100)
            progress_callback(f"Computing normals... {percent}%")

    if progress_callback:
        progress_callback("✅ Structure complete (100%)")

    return vertices, edges, triangles
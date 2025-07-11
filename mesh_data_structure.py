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

def build_mesh_from_stl(file_path):
    mesh = pv.read(file_path)
    points = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # assuming triangular mesh
    
    # Build vertices list
    vertices = [Vertex(coords=points[i], index=i) for i in range(len(points))]
    
    # Build triangles list
    triangles = [Triangle(vertex_indices=face.tolist(), index=i) for i, face in enumerate(faces)]
    
    # Build edges dict to detect duplicates
    edge_dict = {}
    edges = []
    
    for tri in triangles:
        vids = tri.vertex_indices
        # For each edge opposite vertex i, edge i is opposite to vertex i
        # so edges: edge0 opposite vertex0 connects vids[1], vids[2]
        edge_vertices = [
            (vids[1], vids[2]),
            (vids[2], vids[0]),
            (vids[0], vids[1]),
        ]
        for i, (v_start, v_end) in enumerate(edge_vertices):
            # sort vertices so (min,max) to avoid duplicates with reversed order
            key = tuple(sorted((v_start, v_end)))
            if key in edge_dict:
                edge_index = edge_dict[key]
                edges[edge_index].triangles.append(tri.index)
                tri.edge_indices[i] = edge_index
            else:
                edge_index = len(edges)
                edge = Edge(v1=key[0], v2=key[1])
                edge.triangles.append(tri.index)
                edges.append(edge)
                edge_dict[key] = edge_index
                tri.edge_indices[i] = edge_index
    
    # Update valence and triangle indices in vertices
    for tri in triangles:
        for v_idx in tri.vertex_indices:
            vertices[v_idx].valence += 1
            vertices[v_idx].triangle_indices.append(tri.index)
    
    # Compute triangle normals (normalized cross product)
    for tri in triangles:
        p0 = vertices[tri.vertex_indices[0]].coords
        p1 = vertices[tri.vertex_indices[1]].coords
        p2 = vertices[tri.vertex_indices[2]].coords
        edge1 = p1 - p0
        edge2 = p2 - p0
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        tri.normal = n / norm if norm > 0 else np.array([0,0,0])
    
    return vertices, edges, triangles
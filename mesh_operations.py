import numpy as np
import pyvista as pv
from math import acos, degrees

# Laplacian Smoothing: Smooths mesh by averaging vertex positions with neighbors
def laplacian_smoothing(vertices, edges, triangles, iterations=1, lambda_factor=0.5):
    """
    Apply Laplacian smoothing on vertices.
    Moves each vertex toward the average position of its neighbors.
    lambda_factor controls how much the vertex moves per iteration.
    Returns updated vertex list and displacement vectors.
    """
    if len(vertices) > 0 and hasattr(vertices[0], "coords"):
        coords = np.array([v.coords for v in vertices])
    else:
        coords = np.array(vertices)
    
    original_coords = coords.copy()
    V = len(vertices)

    # Build adjacency list: vertex index → list of neighbor indices
    adjacency = [[] for _ in range(V)]
    for edge in edges:
        v1, v2 = edge.v1, edge.v2
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)
        
    # Perform smoothing iterations
    for _ in range(iterations):
        new_coords = coords.copy()
        for i in range(V):
            neighbors = adjacency[i]
            if not neighbors:
                continue
            neighbor_coords = coords[neighbors]
            avg = neighbor_coords.mean(axis=0)
            # Move vertex toward average Position by lambda_factor
            new_coords[i] = coords[i] + lambda_factor * (avg - coords[i])
        coords = new_coords

    # Calculate displacement vectors for analysis or visualization
    diff_vectors = coords - original_coords

    # Update the original vertex objects
    if hasattr(vertices[0], "coords"):
        for i, v in enumerate(vertices):
            v.coords = coords[i]
    else:
        for i in range(len(vertices)):
            vertices[i] = coords[i]
    return vertices, diff_vectors


# Compute shortest distance from a point to the mesh surface
def point_to_mesh_distance(point, vertices, triangles):
    """
    Compute shortest Euclidean distance from a 3D point to the mesh.
    Uses PyVista for accurate nearest-point search.
    """
    points = np.array([v.coords for v in vertices])
    faces = []
    for t in triangles:
        # PyVista expects faces as: [3, v1, v2, v3]
        faces.extend([3] + t.vertex_indices)

    faces = np.array(faces)
    mesh = pv.PolyData(points, faces)
    # Query closest point on the mesh surface
    closest_point_id = mesh.find_closest_point(point)
    closest_point = mesh.points[closest_point_id]
    dist = np.linalg.norm(point - closest_point)
    return dist, closest_point

# Compute angle between adjacent triangles across each edge
def compute_dihedral_angles(edges, triangles):
    """
    Compute dihedral angle (in degrees) between the two triangles sharing each edge.
    Returns a dictionary mapping edge index to angle.
    If an edge is on the boundary (has only one triangle), angle is None.
    """
    angles = {}
    for i, edge in enumerate(edges):
        if len(edge.triangles) != 2:
            angles[i] = None
            continue
        t1_idx, t2_idx = edge.triangles
        n1 = triangles[t1_idx].normal
        n2 = triangles[t2_idx].normal
        dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle_rad = acos(dot)
        angle_deg = degrees(angle_rad)
        angles[i] = angle_deg
    return angles

# MeshOperations class provides static methods for normal computations
class MeshOperations:
    """
    Utility class for computing triangle and vertex normals on a mesh.
    """

    @staticmethod
    def compute_triangle_normals(vertices, triangles):
        """
        Recalculate and store surface normals for each triangle.
        Required after geometry/topology changes.
        """
        for tri in triangles:
            tri.recompute_normal(vertices)

    @staticmethod
    def compute_vertex_normals(vertices, triangles):
        """
        Calculate vertex normals by averaging adjacent triangle normals.
        Useful for smooth shading and visualization.
        """        
        # Reset all vertex normals
        for v in vertices:
            v.normal = np.zeros(3)

        # Accumulate normals from all adjacent triangles
        for tri in triangles:
            for idx in tri.vertex_indices:
                vertices[idx].normal += tri.normal

        # Normalize the final vertex normals
        for v in vertices:
            norm = np.linalg.norm(v.normal)
            if norm > 0:
                v.normal /= norm
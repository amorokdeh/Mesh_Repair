# mesh_operations.py
import numpy as np
import pyvista as pv
from math import acos, degrees

def laplacian_smoothing(vertices, edges, triangles, iterations=1, lambda_factor=0.5):
    """
    Apply Laplacian smoothing on vertices.
    Returns new vertices positions and difference vectors.
    """
    V = len(vertices)
    coords = np.array([v.coords for v in vertices])
    original_coords = coords.copy()

    # Build adjacency list: vertex -> connected vertices
    adjacency = [[] for _ in range(V)]
    for e in edges:
        adjacency[e.v1].append(e.v2)
        adjacency[e.v2].append(e.v1)

    for it in range(iterations):
        new_coords = coords.copy()
        for i in range(V):
            neighbors = adjacency[i]
            if not neighbors:
                continue
            neighbor_coords = coords[neighbors]
            avg = neighbor_coords.mean(axis=0)
            # Move vertex toward average by lambda_factor
            new_coords[i] = coords[i] + lambda_factor * (avg - coords[i])
        coords = new_coords

    # Compute difference vectors
    diff_vectors = coords - original_coords

    # Update vertex coords in place
    for i, v in enumerate(vertices):
        v.coords = coords[i]

    return vertices, diff_vectors


def point_to_mesh_distance(point, vertices, triangles):
    """
    Compute shortest distance from a 3D point to the mesh surface.
    Uses PyVista for geometric queries.
    """
    points = np.array([v.coords for v in vertices])
    faces = []
    for t in triangles:
        # PyVista expects faces as: [3, v1, v2, v3]
        faces.extend([3] + t.vertex_indices)

    faces = np.array(faces)
    mesh = pv.PolyData(points, faces)
    # Find closest point on mesh to the query point
    closest_point_id = mesh.find_closest_point(point)
    closest_point = mesh.points[closest_point_id]
    dist = np.linalg.norm(point - closest_point)
    return dist, closest_point


def compute_dihedral_angles(edges, triangles):
    """
    Compute angle (in degrees) between adjacent triangles for each edge.
    Returns dict edge_index -> angle.
    For boundary edges (only one adjacent triangle), angle = None.
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


def edges_with_large_angle(edges, triangles, threshold_deg=30.0):
    """
    Return list of edge indices where dihedral angle > threshold.
    """
    angles = compute_dihedral_angles(edges, triangles)
    large_angle_edges = [e_idx for e_idx, angle in angles.items() if angle is not None and angle > threshold_deg]
    return large_angle_edges
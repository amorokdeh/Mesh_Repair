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

def triangle_aspect_ratio(p1, p2, p3):
    """Compute the aspect ratio (quality) of a triangle: 4*sqrt(3)*Area / sum(length^2). Higher is better."""
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    area = max(np.sqrt(s * (s - a) * (s - b) * (s - c)), 1e-8)
    return 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2)


def try_edge_flip(edge, vertices, edges, triangles):
    """Try flipping an edge if it improves triangle quality. Returns True if flipped."""
    if len(edge.triangles) != 2:
        return False  # cannot flip boundary edge

    t1_idx, t2_idx = edge.triangles
    t1 = triangles[t1_idx]
    t2 = triangles[t2_idx]

    # Get the 4 unique vertices involved
    v1, v2 = edge.v1, edge.v2
    t1_other = [v for v in t1.vertex_indices if v != v1 and v != v2]
    t2_other = [v for v in t2.vertex_indices if v != v1 and v != v2]

    if len(t1_other) != 1 or len(t2_other) != 1:
        return False  # degenerate triangle
    a = t1_other[0]
    b = t2_other[0]

    # Before flip: triangles are (a,v1,v2) and (b,v2,v1)
    p_a, p_b = vertices[a].coords, vertices[b].coords
    p_v1, p_v2 = vertices[v1].coords, vertices[v2].coords

    old_quality = min(
        triangle_aspect_ratio(p_a, p_v1, p_v2),
        triangle_aspect_ratio(p_b, p_v2, p_v1)
    )
    new_quality = min(
        triangle_aspect_ratio(p_a, p_b, p_v1),
        triangle_aspect_ratio(p_a, p_b, p_v2)
    )

    if new_quality <= old_quality:
        return False  # no improvement

    # ✅ Perform the flip: edge becomes (a, b)
    edge.v1, edge.v2 = a, b

    # Update triangles
    triangles[t1_idx].vertex_indices = [a, b, v1]
    triangles[t2_idx].vertex_indices = [b, a, v2]

    triangles[t1_idx].recompute_normal(vertices)
    triangles[t2_idx].recompute_normal(vertices)

    return True


def beautify_mesh(vertices, edges, triangles):
    """Try flipping all edges to improve triangle quality."""
    flip_count = 0
    for edge in edges:
        if try_edge_flip(edge, vertices, edges, triangles):
            flip_count += 1
    return flip_count
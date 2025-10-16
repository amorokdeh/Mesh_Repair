import numpy as np
import pyvista as pv
from math import acos, degrees
from collections import deque, defaultdict
from mesh_data_structure import (
    build_mesh_from_stl, 
    Vertex, 
    Triangle, 
    Edge
)

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

def taubin_smoothing(vertices, edges, triangles, iterations=10, lambda_factor=0.5, mu_factor=-0.53):
    """
    Apply Taubin smoothing to the mesh.
    Parameters:
        vertices: list of Vertex objects
        edges: list of Edge objects
        triangles: list of Triangle objects
        iterations: number of (lambda+mu) smoothing cycles
        lambda_factor: smoothing factor for Laplacian step
        mu_factor: inverse smoothing factor to reduce shrinkage
    Returns:
        updated vertex list and final displacement vectors
    """
    if len(vertices) == 0:
        return vertices, np.array([])

    V = len(vertices)
    coords = np.array([v.coords for v in vertices])
    original_coords = coords.copy()

    # Build adjacency list
    adjacency = [[] for _ in range(V)]
    for edge in edges:
        v1, v2 = edge.v1, edge.v2
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    def laplacian_step(coords, factor):
        new_coords = coords.copy()
        for i in range(V):
            neighbors = adjacency[i]
            if not neighbors:
                continue
            avg = np.mean(coords[neighbors], axis=0)
            new_coords[i] += factor * (avg - coords[i])
        return new_coords

    for _ in range(iterations):
        coords = laplacian_step(coords, lambda_factor)
        coords = laplacian_step(coords, mu_factor)

    # Update vertex objects
    for i in range(V):
        vertices[i].coords = coords[i]

    diff_vectors = coords - original_coords
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

# --- Utility: get k-ring neighbors using adjacency (list-of-lists) ---
def k_ring_neighbors(adjacency, start_idx, k=2):
    """Return sorted unique vertex indices within k rings of start_idx (including start)."""
    visited = {start_idx}
    frontier = {start_idx}
    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for n in adjacency[v]:
                if n not in visited:
                    visited.add(n)
                    new_frontier.add(n)
        frontier = new_frontier
        if not frontier:
            break
    return sorted(visited)

# --- Detect tubular (cable-like) vertices via local PCA ---
def detect_tubular_regions(vertices, edges, triangles, k_ring=8,
                           eig_ratio_thresh=0.25, radius_thresh=None,
                           min_component_size=30):
    """
    Returns a boolean mask (len(vertices)) where True indicates vertex likely on a thin tubular cable.
    - k_ring: graph-neighborhood size
    - eig_ratio_thresh: threshold for λ2/λ1 and λ3/λ1 to classify 1D/tubular
    - radius_thresh: maximum allowed estimated radius (None disables radius test)
    - min_component_size: remove small components (noise)
    """
    V = len(vertices)
    coords = np.array([v.coords for v in vertices])
    # Build adjacency list
    adjacency = [[] for _ in range(V)]
    for e in edges:
        adjacency[e.v1].append(e.v2)
        adjacency[e.v2].append(e.v1)

    tubular_scores = np.zeros(V, dtype=float)
    tubular_mask = np.zeros(V, dtype=bool)

    # For each vertex compute covariance eigenvalues in neighborhood
    for i in range(V):
        nbrs = k_ring_neighbors(adjacency, i, k=k_ring)
        pts = coords[nbrs]
        if pts.shape[0] < 4:
            continue
        centroid = pts.mean(axis=0)
        P = pts - centroid
        C = (P.T @ P) / max(1, pts.shape[0] - 1)  # covariance
        # symmetric eigendecomposition
        w, _ = np.linalg.eigh(C)  # ascending
        # sort descending
        w = np.sort(w)[::-1]
        # protect against numerical zeros
        if w[0] <= 0:
            continue
        r2 = (w[1] + w[2]) / 2.0  # estimate squared radius (heuristic)
        # tubularity condition: λ2/λ1 and λ3/λ1 small
        ratio2 = w[1] / w[0]
        ratio3 = w[2] / w[0]
        tubular_scores[i] = 1.0 - (ratio2 + ratio3) / 2.0  # higher means more 1D-like
        is_tubular = (ratio2 < eig_ratio_thresh) and (ratio3 < eig_ratio_thresh)
        if is_tubular and radius_thresh is not None:
            radius_est = np.sqrt(max(0.0, r2))
            if radius_est > radius_thresh:
                is_tubular = False
        tubular_mask[i] = bool(is_tubular)

    # Remove tiny components: label connected components on the mask
    def get_components(mask):
        comp_labels = -np.ones(V, dtype=int)
        cur_label = 0
        for idx in range(V):
            if not mask[idx] or comp_labels[idx] != -1:
                continue
            # BFS
            q = deque([idx])
            comp_labels[idx] = cur_label
            while q:
                v = q.popleft()
                for n in adjacency[v]:
                    if mask[n] and comp_labels[n] == -1:
                        comp_labels[n] = cur_label
                        q.append(n)
            cur_label += 1
        return comp_labels, cur_label

    comp_labels, ncomp = get_components(tubular_mask)
    # Count sizes
    sizes = defaultdict(int)
    for lab in comp_labels:
        if lab >= 0:
            sizes[lab] += 1
    # Build final mask filtering small
    final_mask = np.zeros(V, dtype=bool)
    for i in range(V):
        lab = comp_labels[i]
        if lab >= 0 and sizes[lab] >= min_component_size:
            final_mask[i] = True

    return final_mask, tubular_scores

# --- Masked Taubin smoothing (updates only masked vertices) ---
def taubin_smoothing_masked(vertices, edges, triangles, mask,
                            iterations=30, lambda_factor=0.5, mu_factor=-0.53):
    """
    Apply Taubin smoothing but update only vertices with mask==True.
    Non-masked vertices remain fixed (act as boundary constraints).
    Returns updated vertices and displacement vectors (for all vertices).
    """
    if len(vertices) == 0:
        return vertices, np.array([])

    V = len(vertices)
    coords = np.array([v.coords for v in vertices])
    original_coords = coords.copy()

    # Build adjacency list
    adjacency = [[] for _ in range(V)]
    for edge in edges:
        adjacency[edge.v1].append(edge.v2)
        adjacency[edge.v2].append(edge.v1)

    def laplacian_step(coords_arr, factor):
        new_coords = coords_arr.copy()
        # compute Laplacian but only apply to masked vertices
        for i in range(V):
            if not mask[i]:
                continue  # keep fixed
            neighbors = adjacency[i]
            if not neighbors:
                continue
            nbr_coords = coords_arr[neighbors]
            avg = nbr_coords.mean(axis=0)
            new_coords[i] += factor * (avg - coords_arr[i])
        return new_coords

    for _ in range(iterations):
        coords = laplacian_step(coords, lambda_factor)
        coords = laplacian_step(coords, mu_factor)

    # Update vertex objects in place
    for i in range(V):
        vertices[i].coords = coords[i]

    diff_vectors = coords - original_coords
    return vertices, diff_vectors

def vertices_triangles_to_numpy(vertices, triangles):
    vertices_np = np.array([v.coords for v in vertices], dtype=np.float64)
    faces_np = np.array([t.vertex_indices for t in triangles], dtype=np.int32)
    return vertices_np.copy(), faces_np.copy()

# ---------------------------
# Hole detection and filling
# ---------------------------

from collections import defaultdict, deque

def find_boundary_edges(edges):
    """Return list of indices of edges that are boundary edges (only one adjacent triangle)."""
    return [i for i, e in enumerate(edges) if len(e.triangles) == 1]

def build_boundary_adjacency(edges, vertices):
    """
    Build adjacency map of boundary edges:
      vertex_idx -> list of adjacent boundary vertices
    Returns adjacency dict and list of boundary edge indices.
    """
    adjacency = defaultdict(list)
    boundary_edges = find_boundary_edges(edges)
    for e_idx in boundary_edges:
        e = edges[e_idx]
        adjacency[e.v1].append(e.v2)
        adjacency[e.v2].append(e.v1)
    return adjacency, boundary_edges

def extract_ordered_loops_from_adjacency(adjacency):
    """
    Given adjacency of boundary vertices (vertex -> neighbours on boundary),
    extract ordered loops (lists of vertex indices) for each connected component.
    """
    loops = []
    visited = set()

    for start in list(adjacency.keys()):
        if start in visited:
            continue
        # follow the loop
        loop = []
        cur = start
        prev = None
        while True:
            loop.append(cur)
            visited.add(cur)
            neighbors = adjacency[cur]
            # choose neighbor that's not prev (if both exist)
            nxt = None
            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                nxt = neighbors[0]
            else:
                # choose the neighbor different from prev (if any)
                if prev is None:
                    nxt = neighbors[0]
                else:
                    nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
            prev, cur = cur, nxt
            if cur == start or cur in visited:
                break

        # ensure loop is closed and has length > 2
        if len(loop) >= 3:
            # try to ensure distinct ordering by removing a tail if duplicated at end
            # (in pathological adjacency graphs there can be duplicates)
            unique_loop = []
            seen = set()
            for v in loop:
                if v not in seen:
                    unique_loop.append(v)
                    seen.add(v)
            if len(unique_loop) >= 3:
                loops.append(unique_loop)

    return loops

def rebuild_connectivity(vertices, triangles):
    """
    Rebuild edges, triangle indices, vertex triangle lists and valences from scratch.
    Returns (edges, triangles) where triangles' .index are reassigned 0..N-1.
    """
    # Reassign triangle indices
    for i, tri in enumerate(triangles):
        tri.index = i

    edge_dict = {}
    edges = []
    for tri in triangles:
        vids = tri.vertex_indices
        tri.edge_indices = [-1, -1, -1]
        # Skip degenerate triangles
        if len(set(vids)) != 3:
            continue
        edge_vertices = [(vids[0], vids[1]), (vids[1], vids[2]), (vids[2], vids[0])]
        for i_e, (v1, v2) in enumerate(edge_vertices):
            key = tuple(sorted((v1, v2)))
            if key in edge_dict:
                ei = edge_dict[key]
                edges[ei].triangles.append(tri.index)
                tri.edge_indices[i_e] = ei
            else:
                e = Edge(v1=key[0], v2=key[1])
                e.triangles.append(tri.index)
                ei = len(edges)
                edges.append(e)
                edge_dict[key] = ei
                tri.edge_indices[i_e] = ei

    # Reset per-vertex data
    for v in vertices:
        v.valence = 0
        v.triangle_indices = []

    # Accumulate triangle references for vertices
    for tri in triangles:
        if len(set(tri.vertex_indices)) != 3:
            continue
        for vi in tri.vertex_indices:
            vertices[vi].triangle_indices.append(tri.index)
            vertices[vi].valence += 1

    return edges, triangles

def fill_loop_with_fan(vertices, triangles, loop):
    """
    Triangulate the polygon loop by creating fan triangles around loop[0].
    Adds new Triangle objects to triangles in-place. Does not create new vertices.
    """
    if len(loop) < 3:
        return

    base = loop[0]
    # Add triangles (base, loop[i], loop[i+1]) for i = 1..n-2
    start_index = len(triangles)
    for i in range(1, len(loop) - 0 - 1):
        a = base
        b = loop[i]
        c = loop[i + 1]
        # Ensure triangle vertices are distinct
        if a == b or b == c or c == a:
            continue
        tri = Triangle(vertex_indices=[a, b, c], index=start_index)
        triangles.append(tri)
        start_index += 1

def fill_mesh_holes(vertices, edges, triangles, max_loop_size=200):
    """
    Detect boundary loops and fill them with fan triangulation.
    Returns updated (vertices, edges, triangles).
    """

    adjacency, boundary_edges = build_boundary_adjacency(edges, vertices)
    if not adjacency:
        return vertices, edges, triangles

    loops = extract_ordered_loops_from_adjacency(adjacency)
    added_any = False

    for loop in loops:
        if len(loop) <= 2 or len(loop) > max_loop_size:
            continue
        # Fill by fan
        fill_loop_with_fan(vertices, triangles, loop)
        added_any = True

    if not added_any:
        return vertices, edges, triangles

    # Rebuild connectivity
    edges, triangles = rebuild_connectivity(vertices, triangles)

    # --- Fix non-manifold edges ---
    edges_to_remove = [e for e in edges if len(e.triangles) > 2]
    if edges_to_remove:
        bad_tri_indices = set()
        for e in edges_to_remove:
            bad_tri_indices.update(e.triangles)
        triangles = [t for t in triangles if t.index not in bad_tri_indices]
        edges, triangles = rebuild_connectivity(vertices, triangles)

    # Recompute normals
    MeshOperations.compute_triangle_normals(vertices, triangles)
    MeshOperations.compute_vertex_normals(vertices, triangles)

    return vertices, edges, triangles
from collections import defaultdict, deque

def sanity_check_mesh(vertices, edges, triangles, progress_callback=None):
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "holes": [],
        "euler_check": None,
    }

    def report(msg):
        if progress_callback:
            progress_callback(msg)

    # --- 1. Check edges ---
    report("Checking edges...")
    for i, edge in enumerate(edges):
        count_tri = len(edge.triangles)
        if count_tri > 2:
            results["valid"] = False
            results["errors"].append(f"Edge {i} belongs to more than 2 triangles ({count_tri}).")

    boundary_edges = [i for i, e in enumerate(edges) if len(e.triangles) == 1]

    # --- 2. Check valence ---
    report("Checking vertex valence...")
    for i, v in enumerate(vertices):
        if v.valence < 3:
            results["warnings"].append(f"Vertex {i} has valence {v.valence} < 3.")

    # --- 3. Check duplicate points ---
    report("Checking for duplicate vertices...")
    seen_coords = set()
    duplicates = 0
    for v in vertices:
        coord_tuple = tuple(v.coords)
        if coord_tuple in seen_coords:
            duplicates += 1
        else:
            seen_coords.add(coord_tuple)
    if duplicates > 0:
        results["valid"] = False
        results["errors"].append(f"Found {duplicates} duplicate vertex coordinates.")

    # --- 4. Euler's formula ---
    report("Checking Euler characteristic...")
    V = len(vertices)
    E = len(edges)
    F = len(triangles)
    euler_value = V - E + F
    results["euler_check"] = euler_value
    if euler_value != 2:
        results["warnings"].append(f"Euler characteristic V - E + F = {euler_value} (expected 2 for closed manifold).")

    # --- 5. Check holes ---
    report("Checking for holes in the mesh...")
    vertex_to_boundary_edges = {i: 0 for i in range(V)}
    for e_idx in boundary_edges:
        edge = edges[e_idx]
        vertex_to_boundary_edges[edge.v1] += 1
        vertex_to_boundary_edges[edge.v2] += 1

    for v_idx, count in vertex_to_boundary_edges.items():
        if count > 2:
            results["warnings"].append(f"Vertex {v_idx} has {count} boundary edges (max 2 expected).")

    adjacency = defaultdict(list)
    for e_idx in boundary_edges:
        edge = edges[e_idx]
        adjacency[edge.v1].append(edge.v2)
        adjacency[edge.v2].append(edge.v1)

    visited_vertices = set()
    holes = []

    for start_vertex in adjacency:
        if start_vertex in visited_vertices:
            continue
        queue = deque([start_vertex])
        hole_vertices = set()
        hole_edges_count = 0

        while queue:
            v = queue.popleft()
            if v in visited_vertices:
                continue
            visited_vertices.add(v)
            hole_vertices.add(v)
            for nbr in adjacency[v]:
                if nbr not in visited_vertices:
                    queue.append(nbr)
                hole_edges_count += 1

        hole_edges_count = hole_edges_count // 2
        holes.append({
            "vertices_count": len(hole_vertices),
            "edges_count": hole_edges_count
        })

    results["holes"] = holes

    if holes:
        results["warnings"].append(f"Found {len(holes)} hole(s) in the mesh.")

    report("Sanity check completed.")
    return results

def generate_sanity_report(results):
    lines = []

    if results["valid"]:
        lines.append("✅ Mesh passed the basic validity checks.\n")
    else:
        lines.append("❌ Mesh failed validity checks.\n")

    if results["errors"]:
        lines.append("Errors:")
        lines.extend(results["errors"])

    if results["warnings"]:
        lines.append("Warnings:")
        lines.extend(results["warnings"])

    lines.append(f"\nEuler characteristic (V - E + F): {results['euler_check']}")

    if results["holes"]:
        lines.append(f"\nDetected {len(results['holes'])} hole(s):")
        for i, hole in enumerate(results["holes"], 1):
            lines.append(f"  Hole {i}: {hole['edges_count']} edges, {hole['vertices_count']} vertices")

    return "\n".join(lines)

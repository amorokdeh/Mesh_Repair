import numpy as np
from stl import mesh as numpy_stl_mesh 
import json

def save_mesh_to_json(vertices, edges, triangles, filename="mesh_data.json", progress_callback=None):
    data = {
        "vertices": [],
        "edges": [],
        "triangles": []
    }

    total_steps = 3
    current_step = 0

    # Process vertices
    for v in vertices:
        data["vertices"].append({
            "index": v.index,
            "coords": v.coords.tolist(),
            "valence": v.valence,
            "normal": v.normal.tolist() if v.normal is not None else None,
            "triangle_indices": v.triangle_indices
        })
    current_step += 1
    if progress_callback:
        progress_callback(int(current_step / total_steps * 100))

    # Process edges
    for e in edges:
        data["edges"].append({
            "v1": e.v1,
            "v2": e.v2,
            "triangles": e.triangles
        })
    current_step += 1
    if progress_callback:
        progress_callback(int(current_step / total_steps * 100))

    # Process triangles
    for t in triangles:
        data["triangles"].append({
            "index": t.index,
            "vertex_indices": t.vertex_indices,
            "edge_indices": t.edge_indices,
            "normal": t.normal.tolist() if t.normal is not None else None
        })
    current_step += 1
    if progress_callback:
        progress_callback(int(current_step / total_steps * 100))

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    if progress_callback:
        progress_callback(100)

    print(f"✅ Mesh data saved to {filename}")

def save_mesh_to_stl(vertices, triangles, filename):
    # Create numpy arrays for vertices and faces
    stl_data = np.zeros(len(triangles), dtype=numpy_stl_mesh.Mesh.dtype)

    for i, tri in enumerate(triangles):
        v0 = vertices[tri.vertex_indices[0]].coords
        v1 = vertices[tri.vertex_indices[1]].coords
        v2 = vertices[tri.vertex_indices[2]].coords

        stl_data["vectors"][i] = np.array([v0, v1, v2])

        # Optional: Add normal if available
        if tri.normal is not None:
            stl_data["normals"][i] = tri.normal

    mesh = numpy_stl_mesh.Mesh(stl_data)
    mesh.save(filename)
    print(f"✅ STL file saved to {filename}")
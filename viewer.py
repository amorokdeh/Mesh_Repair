import pyvista as pv
import numpy as np

def plot_mesh_from_file(file_path):
    """
    Load mesh from a file and show it.
    """
    mesh = pv.read(file_path)
    plotter = pv.Plotter()
    plotter.set_background('#1e1e1e')
    plotter.add_mesh(
        mesh,
        color='#ccf5ff',
        show_edges=True,
        edge_color='#001f3f',
        line_width=0.5,
        smooth_shading=True,
        specular=0.4,
        specular_power=10,
        opacity=1.0
    )
    plotter.hide_axes()
    plotter.camera_position = 'iso'
    plotter.show()


def plot_mesh_from_data(vertices, triangles, highlight_edges=None, edges=None):
    """
    vertices: list of Vertex objects or Nx3 numeric arrays
    triangles: list of triangle objects or array of triangle indices
    """

    # Convert Vertex objects to Nx3 array
    # Adjust attribute name based on your Vertex class (here assumed 'coords')
    if len(vertices) > 0 and hasattr(vertices[0], "coords"):
        points = np.array([v.coords for v in vertices])
    else:
        # Assume vertices is already Nx3 numeric array
        points = np.array(vertices)

    faces = []
    for t in triangles:
        if hasattr(t, "vertex_indices"):
            indices = t.vertex_indices
        else:
            indices = t
        faces.extend([3] + list(indices))
    faces = np.array(faces)

    mesh = pv.PolyData(points, faces)

    plotter = pv.Plotter()
    plotter.set_background('#1e1e1e')
    plotter.add_mesh(mesh, color='#ccf5ff', show_edges=True, edge_color='#001f3f')

    if highlight_edges and edges:
        for e_idx in highlight_edges:
            edge = edges[e_idx]
            line = pv.Line(points[edge.v1], points[edge.v2])
            plotter.add_mesh(line, color='red', line_width=5)

    plotter.hide_axes()
    plotter.camera_position = 'iso'
    plotter.show()

def plot_mesh_with_highlights(vertices, triangles, highlight_edge_indices, edges):
    points = np.array([v.coords for v in vertices])  # convert objects to coords
    faces = []
    for t in triangles:
        faces.extend([3] + t.vertex_indices)
    faces = np.array(faces)
    mesh = pv.PolyData(points, faces)

    plotter = pv.Plotter()
    plotter.set_background('#1e1e1e')
    plotter.add_mesh(mesh, color='#ccf5ff', show_edges=True, edge_color='#001f3f')

    for e_idx in highlight_edge_indices:
        edge = edges[e_idx]
        line = pv.Line(points[edge.v1], points[edge.v2])
        plotter.add_mesh(line, color='red', line_width=5)

    plotter.hide_axes()
    plotter.camera_position = 'iso'
    plotter.show()

def plot_point_and_closest_on_mesh(vertices, triangles, input_point, closest_point):

    """
    Visualize mesh, input point, and closest point on mesh.
    """

    if len(vertices) > 0 and hasattr(vertices[0], "coords"):
        points = np.array([v.coords for v in vertices])
    else:
        points = np.array(vertices)

    faces = []
    for t in triangles:
        if hasattr(t, "vertex_indices"):
            indices = t.vertex_indices
        else:
            indices = t
        faces.extend([3] + list(indices))
    faces = np.array(faces)

    mesh = pv.PolyData(points, faces)

    plotter = pv.Plotter()
    plotter.set_background('#1e1e1e')

    plotter.add_mesh(mesh, color='#ccf5ff', show_edges=True, edge_color='#001f3f', opacity=0.7)

  # Add input point (red) and closest point (green) with larger radius for visibility
    sphere_radius = 0.01 * np.linalg.norm(points.max(axis=0) - points.min(axis=0))  # scale with model size
    input_sphere = pv.Sphere(radius=sphere_radius, center=input_point)
    closest_sphere = pv.Sphere(radius=sphere_radius, center=closest_point)

    plotter.add_mesh(input_sphere, color='red', label='Input Point')
    plotter.add_mesh(closest_sphere, color='green', label='Closest Point on Mesh')

    # Add a legend
    plotter.add_legend([('Input Point', 'red'), ('Closest Point on Mesh', 'green')])

    # Add orientation aids
    plotter.show_bounds(grid='front', location='outer', color='white')
    plotter.show_axes()

    plotter.camera_position = 'iso'
    plotter.show()
    
def plot_mesh_with_vertex_mask(vertices, triangles, mask, cable_color='red', mesh_color='#ccf5ff'):
    """
    Plot the mesh, coloring masked vertices with cable_color and others with mesh_color.
    vertices: list of Vertex objects or Nx3 numpy array
    triangles: list of Triangle objects or array of triangle indices
    mask: boolean array of length len(vertices)
    """
    import numpy as np
    import pyvista as pv

    # Get Nx3 coords
    if len(vertices) > 0 and hasattr(vertices[0], "coords"):
        points = np.array([v.coords for v in vertices])
    else:
        points = np.array(vertices)

    # Build faces array for PyVista
    faces = []
    for t in triangles:
        if hasattr(t, "vertex_indices"):
            indices = t.vertex_indices
        else:
            indices = t
        faces.extend([3] + list(indices))
    faces = np.array(faces)

    mesh = pv.PolyData(points, faces)

    # Build per-vertex colors
    # Convert cable_color and mesh_color to RGB
    cable_rgb = np.array(pv.Color(cable_color).int_rgb) / 255.0
    mesh_rgb = np.array(pv.Color(mesh_color).int_rgb) / 255.0

    colors = np.zeros((len(vertices), 3))
    colors[mask] = cable_rgb
    colors[~mask] = mesh_rgb

    mesh.point_data['colors'] = colors

    plotter = pv.Plotter()
    plotter.set_background('#1e1e1e')
    plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=True, edge_color='#001f3f')
    plotter.hide_axes()
    plotter.camera_position = 'iso'
    plotter.show()

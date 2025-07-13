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

def plot_mesh_with_highlights(vertices, triangles, highlight_edge_indices):
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
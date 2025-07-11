import pyvista as pv

def plot_mesh_process(file_path):
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
    plotter.show(title=f"STL Viewer: {file_path}")
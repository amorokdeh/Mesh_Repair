import pyvista as pv

def load_stl(file_path):
    return pv.read(file_path)

def repair_mesh(mesh):
    # Placeholder for mesh repair if needed
    return mesh
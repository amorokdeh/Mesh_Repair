import tkinter as tk
from tkinter import filedialog, messagebox
from multiprocessing import Process
import mesh_io
import viewer

def gui_load_and_view():
    root = tk.Tk()
    root.title("STL Viewer Launcher")
    root.geometry("350x150")

    menubar = tk.Menu(root)
    root.config(menu=menubar)

    info_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Mesh Info", menu=info_menu)
    info_menu.add_command(label="No mesh loaded", state='disabled')

    status_var = tk.StringVar()
    status_var.set("No mesh loaded")
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 10))
    status_label.pack(pady=(10, 0))

    def update_mesh_info(mesh):
        info_menu.delete(0, 'end')
        vertices = mesh.n_points
        triangles = mesh.n_cells
        info_menu.add_command(label=f"Vertices: {vertices}", state='disabled')
        info_menu.add_command(label=f"Triangles: {triangles}", state='disabled')

        status_var.set(f"Vertices: {vertices} | Triangles: {triangles}")

    def on_load_click():
        file_path = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[("STL Files", "*.stl")]
        )
        if not file_path:
            messagebox.showinfo("No file selected", "Please select an STL file.")
            return
        try:
            mesh = mesh_io.load_stl(file_path)
            mesh = mesh_io.repair_mesh(mesh)
            update_mesh_info(mesh)

            # Launch PyVista viewer in a separate process
            p = Process(target=viewer.plot_mesh_process, args=(file_path,))
            p.daemon = True
            p.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load/display mesh:\n{e}")

    btn_load = tk.Button(root, text="Load STL File", command=on_load_click, height=2, width=20)
    btn_load.pack(expand=True)

    root.mainloop()
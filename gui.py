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

    def update_mesh_info(vertices, edges, triangles):
        info_menu.delete(0, 'end')
        info_menu.add_command(label=f"Vertices: {len(vertices)}", state='disabled')
        info_menu.add_command(label=f"Edges: {len(edges)}", state='disabled')
        info_menu.add_command(label=f"Triangles: {len(triangles)}", state='disabled')

        status_var.set(f"Vertices: {len(vertices)} | Edges: {len(edges)} | Triangles: {len(triangles)}")

    def on_load_click():
        file_path = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[("STL Files", "*.stl")]
        )
        if not file_path:
            messagebox.showinfo("No file selected", "Please select an STL file.")
            return
        try:
            vertices, edges, triangles = mesh_io.load_stl(file_path)
            # repair_mesh if needed
            vertices, edges, triangles = mesh_io.repair_mesh((vertices, edges, triangles))
            update_mesh_info(vertices, edges, triangles)

            # Launch PyVista viewer in a separate process (pass file path or mesh)
            p = Process(target=viewer.plot_mesh_process, args=(file_path,))
            p.daemon = True
            p.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load/display mesh:\n{e}")

    btn_load = tk.Button(root, text="Load STL File", command=on_load_click, height=2, width=20)
    btn_load.pack(expand=True)

    root.mainloop()
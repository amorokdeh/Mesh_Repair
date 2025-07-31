import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tkinter.simpledialog as sd

from multiprocessing import Process
import threading
import viewer
import time
import copy
from mesh_export import save_mesh_to_json, save_mesh_to_stl
from mesh_data_structure import build_mesh_from_stl
from mesh_sanity_check import sanity_check_mesh, generate_sanity_report
from mesh_operations import laplacian_smoothing, point_to_mesh_distance
from mesh_operations import  compute_dihedral_angles
from mesh_operations import MeshOperations
from tkinter import simpledialog


def gui_load_and_view():
    root = tk.Tk()
    root.title("STL Viewer Launcher")
    root.geometry("400x200")

    menubar = tk.Menu(root)
    root.config(menu=menubar)

    info_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Mesh Info", menu=info_menu)
    info_menu.add_command(label="No mesh loaded", state='disabled')

    action_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Actions", menu=action_menu)
    action_menu.add_command(label="Build Data Structure", state='disabled', command=lambda: build_structure())
    action_menu.add_command(label="Export Mesh", state='disabled', command=lambda: export_mesh())
    action_menu.add_command(label="Sanity Check Mesh", state='disabled', command=lambda: sanity_check())
    action_menu.add_command(label="Laplacian Smoothing", state='disabled', command=lambda: laplacian_smoothing_gui())
    action_menu.add_command(label="Compute Dihedral Angles", state='disabled', command=lambda: compute_dihedral_angles())
    action_menu.add_command(label="Compute Point-Mesh Distance", state='disabled', command=lambda: compute_point_mesh_distance_gui())

    status_var = tk.StringVar()
    status_var.set("No mesh loaded")
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 10))
    status_label.pack(pady=(10, 0))

    app_state = {
        "vertices": None,
        "edges": None,
        "triangles": None,
        "file_path": None
    }

    def update_mesh_info(vertices, edges, triangles):
        info_menu.delete(0, 'end')
        info_menu.add_command(label=f"Vertices: {len(vertices)}", state='disabled')
        info_menu.add_command(label=f"Edges: {len(edges)}", state='disabled')
        info_menu.add_command(label=f"Triangles: {len(triangles)}", state='disabled')

        status_var.set(f"Vertices: {len(vertices)} | Edges: {len(edges)} | Triangles: {len(triangles)}")

    def build_structure():
        def report_progress(msg):
            status_var.set(msg)
            root.update_idletasks()

        try:
            action_menu.entryconfig("Build Data Structure", state="disabled")
            report_progress("⏳ Building Data Structure...")

            start = time.time()
            vertices, edges, triangles = build_mesh_from_stl(
                app_state["file_path"],
                progress_callback=lambda percent: report_progress(f"🔧 Building... {percent}")
            )
            end = time.time()
            
            app_state["vertices"] = vertices
            app_state["edges"] = edges
            app_state["triangles"] = triangles

            MeshOperations.compute_triangle_normals(vertices, triangles)
            MeshOperations.compute_vertex_normals(vertices, triangles)

            update_mesh_info(vertices, edges, triangles)
            messagebox.showinfo("Success", "Data Structure created successfully.\n" f"⏱️ Build time: {end - start:.6f} seconds")

            action_menu.entryconfig("Export Mesh", state="normal")
            action_menu.entryconfig("Sanity Check Mesh", state="normal")
            action_menu.entryconfig("Laplacian Smoothing", state="normal")
            action_menu.entryconfig("Compute Dihedral Angles", state="normal") 
            action_menu.entryconfig("Compute Point-Mesh Distance", state="normal")

            report_progress("✅ Data Structure ready")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to build structure:\n{e}")
            status_var.set("❌ Structure build failed")
            action_menu.entryconfig("Build Data Structure", state="normal")

    def export_mesh():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("STL files", "*.stl")],
            title="Export Mesh As"
        )
        if not file_path:
            return  # User canceled

        def run_export():
            def progress_cb(percent):
                status_var.set(f"📤 Exporting mesh... {percent}%")
                root.update_idletasks()

            try:
                if file_path.endswith(".json"):
                    save_mesh_to_json(
                        app_state["vertices"],
                        app_state["edges"],
                        app_state["triangles"],
                        filename=file_path,
                        progress_callback=progress_cb
                    )
                    messagebox.showinfo("Exported", f"Mesh exported to '{file_path}'.")
                elif file_path.endswith(".stl"):
                    save_mesh_to_stl(
                        app_state["vertices"],
                        app_state["triangles"],
                        file_path
                    )
                    messagebox.showinfo("Exported", f"Mesh exported to '{file_path}'.")
                else:
                    raise ValueError("Unsupported file extension.")

                status_var.set("✅ Export completed")

            except Exception as e:
                status_var.set("❌ Export failed")
                messagebox.showerror("Error", f"Failed to export mesh:\n{e}")

        threading.Thread(target=run_export, daemon=True).start()


    def load_mesh():
        file_path = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[("STL Files", "*.stl")]
        )
        if not file_path:
            messagebox.showinfo("No file selected", "Please select an STL file.")
            return

        def load():
            try:
                def report_status(msg):
                    status_var.set(msg)
                    root.update_idletasks()

                report_status("⏳ Loading STL file...")
                app_state["file_path"] = file_path

                # Launch viewer only
                p = Process(target=viewer.plot_mesh_from_file, args=(file_path,))
                p.daemon = True
                p.start()

                report_status("✅ STL file loaded. Ready to build structure.")
                
                # Enable buttons
                action_menu.entryconfig("Build Data Structure", state="normal")
                action_menu.entryconfig("Export Mesh", state="disabled")
                action_menu.entryconfig("Sanity Check Mesh", state="disabled") 

                # Hide Load button
                btn_load.config(state="disabled")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load/display mesh:\n{e}")
                status_var.set("❌ Load failed")

        threading.Thread(target=load).start()

    def sanity_check():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        def progress_update(msg):
            status_var.set(f"🛠️ {msg}")
            root.update_idletasks()

        def run_check():
            try:
                start = time.time()
                results = sanity_check_mesh(
                    app_state["vertices"],
                    app_state["edges"],
                    app_state["triangles"],
                    progress_callback=progress_update
                )
                end = time.time()
                runtime = end - start
                msg = generate_sanity_report(results)
                msg += f"\n\nSanity check runtime: {runtime:.6f} seconds"


                status_var.set("✅ Sanity check done.")
                messagebox.showinfo("Sanity Check Result", msg)

                # Save the report to a file
                try:
                    with open("sanity_check_report.txt", "w", encoding="utf-8") as f:
                        f.write(msg)
                except Exception as e:
                    messagebox.showwarning("Export Failed", f"Could not save report:\n{e}")

            except Exception as e:
                status_var.set("❌ Sanity check error")
                messagebox.showerror("Error", f"Sanity check failed:\n{e}")


        threading.Thread(target=run_check, daemon=True).start()

    def laplacian_smoothing_gui():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        # Simple input dialogs for iterations and lambda
        iterations = sd.askinteger("Laplacian Smoothing", "Number of iterations:", minvalue=1, maxvalue=100, initialvalue=1)
        if iterations is None:
            return
        lambda_factor = sd.askfloat("Laplacian Smoothing", "Lambda factor (0 to 1):", minvalue=0.0, maxvalue=1.0, initialvalue=0.5)
        if lambda_factor is None:
            return

        def run_smoothing():
            status_var.set("🛠️ Applying Laplacian smoothing...")
            root.update_idletasks()

            start = time.time()
            if "original_vertices" not in app_state:
                app_state["original_vertices"] = copy.deepcopy(app_state["vertices"])
            
            vertices, diff_vectors = laplacian_smoothing(
                app_state["vertices"],
                app_state["edges"],
                app_state["triangles"],
                iterations=iterations,
                lambda_factor=lambda_factor
            )

            end = time.time()
            runtime = end - start

            app_state["vertices"] = vertices  # update app state

            moved_distances = np.linalg.norm(diff_vectors, axis=1)
            max_move = moved_distances.max()
           
            messagebox.showinfo(
                "Laplacian Smoothing", 
                f"Smoothing done.\nMax vertex move distance: {max_move:.4f}\n" 
                f"Runtime: {runtime:.6f} seconds"
                )

            status_var.set("✅ Smoothing complete.")

            # Launch viewer with updated mesh
            def show_updated():
                viewer.plot_mesh_from_data(app_state["vertices"], app_state["triangles"])

            threading.Thread(target=show_updated, daemon=True).start()


        threading.Thread(target=run_smoothing, daemon=True).start()

    def compute_point_mesh_distance_gui():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        point_str = simpledialog.askstring("Input Point", "Enter point coordinates as x,y,z:")
        if point_str is None:
            return

        try:
            point = np.array([float(c.strip()) for c in point_str.split(",")])
            if point.shape != (3,):
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid Input", "Please enter a valid 3D point as x,y,z.")
            return

        dist, closest_point = point_to_mesh_distance(point, app_state["vertices"], app_state["triangles"])
        viewer.plot_point_and_closest_on_mesh(app_state["vertices"], app_state["triangles"], point, closest_point)

        messagebox.showinfo(
            "Point-Mesh Distance",
            f"Shortest distance: {dist:.6f}\n"
            f"Closest point on mesh: {closest_point}"
        )
        status_var.set("✅ Point-mesh distance computed.")

    btn_load = tk.Button(root, text="Load STL File", command=load_mesh, height=2, width=20)
    btn_load.pack(expand=True)


    def compute_dihedral_angles():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        from mesh_operations import compute_dihedral_angles

        status_var.set("🛠️ Computing dihedral angles...")
        root.update_idletasks()

        MeshOperations.compute_triangle_normals(app_state["vertices"], app_state["triangles"])

        angles = compute_dihedral_angles(app_state["edges"], app_state["triangles"])

        if not angles:
            messagebox.showinfo("Dihedral Angles", "No angles could be computed.")
            status_var.set("✅ Computation complete.")
            return

        # Basic statistics
        angle_values = [a for a in angles.values() if a is not None]
        if not angle_values:
            messagebox.showinfo("Dihedral Angles", "No non-boundary edges found.")
            status_var.set("✅ Computation complete.")
            return

        angle_array = np.array(angle_values)
        msg = (
            f"Computed angles for {len(angle_array)} edges.\n"
            f"Min: {angle_array.min():.2f}°\n"
            f"Max: {angle_array.max():.2f}°\n"
            f"Mean: {angle_array.mean():.2f}°\n"
            f"Std: {angle_array.std():.2f}°"
        )
        messagebox.showinfo("Dihedral Angles Statistics", msg)
        status_var.set("✅ Dihedral angles computed.")

    root.mainloop()
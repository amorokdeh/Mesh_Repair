import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from multiprocessing import Process
import threading
import mesh_io
import viewer
from mesh_export import save_mesh_to_json, save_mesh_to_stl
from mesh_data_structure import build_mesh_from_stl
from mesh_sanity_check import sanity_check_mesh, generate_sanity_report
from mesh_operations import laplacian_smoothing, point_to_mesh_distance, edges_with_large_angle

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
    action_menu.add_command(label="Highlight Sharp Edges", state='disabled', command=lambda: highlight_sharp_edges())
    action_menu.add_command(label="BeautiFill Mesh", state='disabled', command=lambda: beautify_mesh_gui())

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

            vertices, edges, triangles = build_mesh_from_stl(
                app_state["file_path"],
                progress_callback=lambda percent: report_progress(f"🔧 Building... {percent}")
            )

            app_state["vertices"] = vertices
            app_state["edges"] = edges
            app_state["triangles"] = triangles

            update_mesh_info(vertices, edges, triangles)
            messagebox.showinfo("Success", "Data Structure created successfully.")

            action_menu.entryconfig("Export Mesh", state="normal")
            action_menu.entryconfig("Sanity Check Mesh", state="normal")
            action_menu.entryconfig("Laplacian Smoothing", state="normal")
            action_menu.entryconfig("Highlight Sharp Edges", state="normal")
            action_menu.entryconfig("BeautiFill Mesh", state="normal")



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
                action_menu.entryconfig("BeautiFill Mesh", state="disabled")

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
                results = sanity_check_mesh(
                    app_state["vertices"],
                    app_state["edges"],
                    app_state["triangles"],
                    progress_callback=progress_update
                )

                msg = generate_sanity_report(results)

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
        import tkinter.simpledialog as sd
        iterations = sd.askinteger("Laplacian Smoothing", "Number of iterations:", minvalue=1, maxvalue=100, initialvalue=1)
        if iterations is None:
            return
        lambda_factor = sd.askfloat("Laplacian Smoothing", "Lambda factor (0 to 1):", minvalue=0.0, maxvalue=1.0, initialvalue=0.5)
        if lambda_factor is None:
            return

        def run_smoothing():
            from mesh_operations import laplacian_smoothing

            status_var.set("🛠️ Applying Laplacian smoothing...")
            root.update_idletasks()

            vertices, diff_vectors = laplacian_smoothing(
                app_state["vertices"],
                app_state["edges"],
                app_state["triangles"],
                iterations=iterations,
                lambda_factor=lambda_factor
            )

            app_state["vertices"] = vertices  # update app state

            moved_distances = np.linalg.norm(diff_vectors, axis=1)
            max_move = moved_distances.max()

            messagebox.showinfo("Laplacian Smoothing", f"Smoothing done.\nMax vertex move distance: {max_move:.4f}")

            status_var.set("✅ Smoothing complete.")

            # Launch viewer with updated mesh
            def show_updated():
                viewer.plot_mesh_from_data(app_state["vertices"], app_state["triangles"])

            threading.Thread(target=show_updated, daemon=True).start()


        threading.Thread(target=run_smoothing, daemon=True).start()

    def highlight_sharp_edges():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        import tkinter.simpledialog as sd
        threshold = sd.askfloat("Highlight Sharp Edges", "Angle threshold (degrees):", minvalue=0, maxvalue=180, initialvalue=30)
        if threshold is None:
            return

        from mesh_operations import edges_with_large_angle
        sharp_edges = edges_with_large_angle(app_state["edges"], app_state["triangles"], threshold_deg=threshold)

        if not sharp_edges:
            messagebox.showinfo("Sharp Edges", "No edges found with angle above threshold.")
            return

        status_var.set(f"🛠️ Highlighting {len(sharp_edges)} edges with angle > {threshold}°")

        # Launch PyVista viewer showing highlighted edges
        def run_view():
            import viewer
            viewer.plot_mesh_with_highlights(app_state["vertices"], app_state["triangles"], sharp_edges, app_state["edges"])

        threading.Thread(target=run_view, daemon=True).start()

    def beautify_mesh_gui():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        def run_beautify():
            from mesh_operations import beautify_mesh

            status_var.set("🛠️ Beautifying mesh (edge flips)...")
            root.update_idletasks()

            flip_count = beautify_mesh(
                app_state["vertices"],
                app_state["edges"],
                app_state["triangles"]
            )

            status_var.set(f"✅ Beautification complete. {flip_count} edges flipped.")
            messagebox.showinfo("BeautiFill Result", f"Beautification done.\nEdges flipped: {flip_count}")

            # Show updated mesh
            def show_updated():
                viewer.plot_mesh_from_data(app_state["vertices"], app_state["triangles"])

            threading.Thread(target=show_updated, daemon=True).start()

        threading.Thread(target=run_beautify, daemon=True).start()

    btn_load = tk.Button(root, text="Load STL File", command=load_mesh, height=2, width=20)
    btn_load.pack(expand=True)

    root.mainloop()
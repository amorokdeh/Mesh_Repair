import tkinter as tk
from tkinter import filedialog, messagebox
from multiprocessing import Process
import threading
import mesh_io
import viewer
from mesh_export import save_mesh_to_json
from mesh_data_structure import build_mesh_from_stl
from mesh_sanity_check import sanity_check_mesh, generate_sanity_report

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
    action_menu.add_command(label="Build Datensstruktur", state='disabled', command=lambda: build_structure())
    action_menu.add_command(label="Export Mesh", state='disabled', command=lambda: export_mesh())
    # New Sanity Check menu item, initially disabled
    action_menu.add_command(label="Sanity Check Mesh", state='disabled', command=lambda: sanity_check())

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
            action_menu.entryconfig("Build Datensstruktur", state="disabled")
            report_progress("⏳ Building Datensstruktur...")

            vertices, edges, triangles = build_mesh_from_stl(
                app_state["file_path"],
                progress_callback=lambda percent: report_progress(f"🔧 Building... {percent}")
            )

            app_state["vertices"] = vertices
            app_state["edges"] = edges
            app_state["triangles"] = triangles

            update_mesh_info(vertices, edges, triangles)
            messagebox.showinfo("Success", "Datensstruktur created successfully.")

            action_menu.entryconfig("Export Mesh", state="normal")
            # Enable sanity check menu
            action_menu.entryconfig("Sanity Check Mesh", state="normal")

            report_progress("✅ Datensstruktur ready")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to build structure:\n{e}")
            status_var.set("❌ Structure build failed")
            action_menu.entryconfig("Build Datensstruktur", state="normal")

    def export_mesh():
        if not app_state["vertices"]:
            messagebox.showwarning("No Data", "Please build the structure first.")
            return

        def run_export():
            def progress_cb(percent):
                status_var.set(f"📤 Exporting mesh... {percent}%")
                root.update_idletasks()

            try:
                save_mesh_to_json(
                    app_state["vertices"],
                    app_state["edges"],
                    app_state["triangles"],
                    filename="exported_mesh.json",
                    progress_callback=progress_cb
                )
                status_var.set("✅ Export completed")
                messagebox.showinfo("Exported", "Mesh exported to 'exported_mesh.json'.")
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
                p = Process(target=viewer.plot_mesh_process, args=(file_path,))
                p.daemon = True
                p.start()

                report_status("✅ STL file loaded. Ready to build structure.")
                
                # Enable buttons
                action_menu.entryconfig("Build Datensstruktur", state="normal")
                action_menu.entryconfig("Export Mesh", state="disabled")
                action_menu.entryconfig("Sanity Check Mesh", state="disabled")  # Disabled until structure built

                # Hide Load button
                btn_load.pack_forget()

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

    btn_load = tk.Button(root, text="Load STL File", command=load_mesh, height=2, width=20)
    btn_load.pack(expand=True)

    root.mainloop()
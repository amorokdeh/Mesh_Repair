import sys
import copy
import threading
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QAction, 
    QFileDialog, 
    QMessageBox, 
    QInputDialog,
    QTextEdit, 
    QDockWidget,
    QProgressBar, 
    QWidget, 
    QVBoxLayout, 
    QLabel
)
from PyQt5.QtCore import (Qt, 
    QTimer, 
    pyqtSignal,
    QObject
)
from mesh_data_structure import (
    build_mesh_from_stl, 
    Vertex, 
    Triangle, 
    Edge
)
from mesh_sanity_check import (
    sanity_check_mesh, 
    generate_sanity_report
)
from mesh_operations import (
    MeshOperations, 
    laplacian_smoothing, 
    taubin_smoothing, 
    vertices_triangles_to_numpy,
    compute_dihedral_angles, 
    point_to_mesh_distance, 
    detect_tubular_regions,
    taubin_smoothing_masked
)
from mesh_export import (
    save_mesh_to_json, 
    save_mesh_to_stl
)

try:
    import QEM
except Exception:
    QEM = None
    print("Warning: QEM extension not found; simplification will be unavailable.")

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    report_ready = pyqtSignal(str)
    update_preview = pyqtSignal(object)
    mesh_update = pyqtSignal(object, object)
    show_message = pyqtSignal(str, str)

class MeshApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Repair Tool")
        self.resize(1600, 1000)

        # --- Embedded 3D viewer ---
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.setCentralWidget(self.plotter)

        # Signals connections
        self.signals = WorkerSignals()
        self.signals.mesh_update.connect(self.update_mesh_in_plotter)
        self.signals.show_message.connect(lambda title, msg: QMessageBox.information(self, title, msg))
        self.signals.log.connect(self.log)

        # --- container for log + progress bar ---
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")  # dark background

        dock_container = QWidget()
        dock_layout = QVBoxLayout()
        dock_layout.setContentsMargins(2, 2, 2, 2)
        dock_layout.setSpacing(4)
        dock_container.setLayout(dock_layout)
        dock_container.setStyleSheet("background-color: #1e1e1e;")  # dark background for container

        dock_layout.addWidget(self.log_panel)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #555555;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
            }
        """)
        self.progress.hide()
        dock_layout.addWidget(self.progress)

        dock = QDockWidget("Logs", self)
        dock.setWidget(dock_container)
        dock.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")  # dark dock background
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # --- Menu bar ---
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #333333;
            }
        """)

        # --- Status bar for mesh info ---
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self.mesh_info_label = QLabel("Vertices: 0 | Edges: 0 | Faces: 0")
        self.status_bar.addPermanentWidget(self.mesh_info_label)

        file_menu = menubar.addMenu("File")
        self.action_load = QAction("Load STL Mesh", self)
        self.action_load.triggered.connect(self.load_stl)
        file_menu.addAction(self.action_load)

        self.action_export = QAction("Export Mesh", self)
        self.action_export.triggered.connect(self.export_mesh)
        self.action_export.setEnabled(False)
        file_menu.addAction(self.action_export)

        actions_menu = menubar.addMenu("Actions")
        self.action_build = QAction("Build Data Structure", self)
        self.action_build.setEnabled(False)
        self.action_build.triggered.connect(self.build_structure)
        actions_menu.addAction(self.action_build)

        self.action_sanity = QAction("Sanity Check Mesh", self)
        self.action_sanity.setEnabled(False)
        self.action_sanity.triggered.connect(self.sanity_check)
        actions_menu.addAction(self.action_sanity)

        self.action_lap = QAction("Laplacian Smoothing", self)
        self.action_lap.setEnabled(False)
        self.action_lap.triggered.connect(self.laplacian_smoothing_gui)
        actions_menu.addAction(self.action_lap)

        self.action_taubin = QAction("Taubin Smoothing", self)
        self.action_taubin.setEnabled(False)
        self.action_taubin.triggered.connect(self.taubin_smoothing_gui)
        actions_menu.addAction(self.action_taubin)

        self.action_cables = QAction("Detect And Smooth Cables", self)
        self.action_cables.setEnabled(False)
        self.action_cables.triggered.connect(self.detect_and_smooth_cables)
        actions_menu.addAction(self.action_cables)

        self.action_dihedral = QAction("Compute Dihedral Angles", self)
        self.action_dihedral.setEnabled(False)
        self.action_dihedral.triggered.connect(self.compute_dihedral_angles)
        actions_menu.addAction(self.action_dihedral)

        self.action_pointdist = QAction("Compute Point-Mesh Distance", self)
        self.action_pointdist.setEnabled(False)
        self.action_pointdist.triggered.connect(self.compute_point_mesh_distance_gui)
        actions_menu.addAction(self.action_pointdist)

        self.action_qem = QAction("Simplify Mesh (QEM)", self)
        self.action_qem.setEnabled(False and (QEM is not None))
        self.action_qem.triggered.connect(self.qem_simplify_mesh)
        actions_menu.addAction(self.action_qem)

        # --- App state (kept same semantics as your Tk app) ---
        self.state = {
            "file_path": None,
            "vertices": None,
            "edges": None,
            "triangles": None,
            "original_vertices": None,
        }

    # --- Update status bar when mesh changes ---
    def update_mesh_info(self, vertices, edges, faces):
        self.mesh_info_label.setText(
            f"Vertices: {len(vertices)} | Edges: {len(edges)} | Faces: {len(faces)}"
        )

    def update_mesh_in_plotter(self, pts, colors):
        faces = []
        for t in self.state["triangles"]:
            faces.extend([3] + list(t.vertex_indices))
        mesh = pv.PolyData(pts, np.array(faces))
        mesh.point_data["colors"] = colors
        self.plotter.clear()
        self.plotter.add_mesh(mesh, scalars="colors", rgb=True, show_edges=True, edge_color="#001f3f")
        self.plotter.reset_camera()

        # --- Update mesh info ---
        self.update_mesh_info(
        self.state["vertices"], 
        self.state.get("edges", []),
        self.state["triangles"]
    )

    # ---- Helpers ----
    def log(self, msg: str):
        self.log_panel.append(msg)
        # keep also to stdout for devs
        print(msg)

    def set_actions_enabled_for_loaded(self, enabled: bool):
        self.action_build.setEnabled(enabled)
        self.action_export.setEnabled(False)
        self.action_sanity.setEnabled(False)
        self.action_lap.setEnabled(False)
        self.action_taubin.setEnabled(False)
        self.action_cables.setEnabled(False)
        self.action_dihedral.setEnabled(False)
        self.action_pointdist.setEnabled(False)
        self.action_qem.setEnabled(False and (QEM is not None))

    def set_actions_enabled_for_built(self, enabled: bool):
        self.action_export.setEnabled(enabled)
        self.action_sanity.setEnabled(enabled)
        self.action_lap.setEnabled(enabled)
        self.action_taubin.setEnabled(enabled)
        self.action_cables.setEnabled(enabled)
        self.action_dihedral.setEnabled(enabled)
        self.action_pointdist.setEnabled(enabled)
        self.action_qem.setEnabled(enabled and (QEM is not None))

    def draw_mesh_from_vertices_triangles(self):
        """Render current state vertices/triangles in the embedded plotter."""
        v = self.state["vertices"]
        t = self.state["triangles"]
        if not v or not t:
            return

        # Build numpy arrays
        points = np.array([vv.coords for vv in v])
        faces = []
        for tri in t:
            faces.extend([3] + list(tri.vertex_indices))
        faces = np.array(faces)

        mesh = pv.PolyData(points, faces)
        self.plotter.clear()
        self.plotter.set_background("#1e1e1e")
        self.plotter.add_mesh(
            mesh,
            color="#ccf5ff",
            show_edges=True,
            edge_color="#001f3f",
            line_width=0.5,
            smooth_shading=True,
        )
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()

    # ---- Menu actions ----
    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
        if not path:
            return
        self.state["file_path"] = path
        self.log(f"📂 Loading STL: {path}")

        try:
            mesh = pv.read(path)
            self.plotter.clear()
            self.plotter.set_background("#1e1e1e")
            self.plotter.add_mesh(
                mesh, color="#ccf5ff", show_edges=True, edge_color="#001f3f",
                line_width=0.5, smooth_shading=True
            )
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
            self.log("✅ STL loaded. Ready to build data structure.")
            self.set_actions_enabled_for_loaded(True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self.log(f"❌ Load failed: {e}")

    def export_mesh(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh", "", "JSON (*.json);;STL (*.stl)"
        )
        if not path:
            return

        def worker():
            try:
                if path.lower().endswith(".json"):
                    def progress_cb(pct):
                        self.log(f"📤 Exporting JSON... {pct}%")
                    save_mesh_to_json(
                        self.state["vertices"], self.state["edges"], self.state["triangles"],
                        filename=path, progress_callback=progress_cb
                    )
                elif path.lower().endswith(".stl"):
                    self.log("📤 Exporting STL...")
                    save_mesh_to_stl(self.state["vertices"], self.state["triangles"], path)
                else:
                    raise ValueError("Unsupported extension.")
                self.log("✅ Export completed.")
            except Exception as e:
                self.log(f"❌ Export failed: {e}")
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Export Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def build_structure(self):
        if not self.state["file_path"]:
            QMessageBox.information(self, "Info", "Please load an STL first.")
            return

        self.progress.show()
        self.progress.setValue(0)
        self.action_build.setEnabled(False)

        # Use the existing WorkerSignals
        signals = WorkerSignals()

        # Connect signals to GUI updates
        signals.progress.connect(self.progress.setValue)
        signals.log.connect(self.log)
        signals.finished.connect(lambda: [
            self.progress.hide(),
            self.action_build.setEnabled(True),
            self.set_actions_enabled_for_built(True),
            self.update_mesh_info(
                self.state["vertices"],
                self.state["edges"],
                self.state["triangles"]
            )
        ])
        signals.error.connect(lambda msg: [
            self.progress.hide(),
            self.action_build.setEnabled(True),
            QMessageBox.critical(self, "Build Error", msg),
            self.log(f"❌ Build failed: {msg}")
        ])

        # Draw mesh in plotter when finished
        signals.finished.connect(self.draw_mesh_from_vertices_triangles)

        def worker():
            import time
            t0 = time.time()
            try:
                # Callback for mesh building (only sends integer percent)
                def callback(pct_str: str):
                    try:
                        pct = int(pct_str)
                        signals.progress.emit(pct)
                    except ValueError:
                        pass  # ignore non-integer messages

                # Build the mesh structure
                vertices, edges, triangles = build_mesh_from_stl(
                    self.state["file_path"],
                    progress_callback=callback
                )

                # Compute normals
                MeshOperations.compute_triangle_normals(vertices, triangles)
                MeshOperations.compute_vertex_normals(vertices, triangles)

                # Update app state
                self.state["vertices"] = vertices
                self.state["edges"] = edges
                self.state["triangles"] = triangles

                t1 = time.time()
                signals.log.emit(f"✅ Data structure ready (time: {t1 - t0:.3f}s)")

            except Exception as e:
                signals.error.emit(str(e))

            finally:
                signals.finished.emit()

        threading.Thread(target=worker, daemon=True).start()

    def sanity_check(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        self.progress.show()
        self.progress.setValue(0)
        self.log("🧪 Running sanity check...")

        signals = WorkerSignals()
        # Add a custom signal for sending the report
        if not hasattr(signals, "report_ready"):
            signals.report_ready = pyqtSignal(str)  

        # Connect signals
        signals.progress.connect(self.progress.setValue)
        signals.finished.connect(lambda: self.progress.hide())
        signals.finished.connect(lambda: self.log("✅ Sanity check complete."))
        signals.log.connect(self.log)
        signals.report_ready.connect(self._show_sanity_report)  # will show the report

        def worker():
            import time
            t0 = time.time()
            try:
                results = sanity_check_mesh(
                    self.state["vertices"], self.state["edges"], self.state["triangles"],
                    progress_callback=lambda msg: signals.log.emit(msg)
                )
                t1 = time.time()
                report_msg = generate_sanity_report(results)
                report_msg += f"\n\nSanity check runtime: {t1 - t0:.6f} seconds"

                # Send the report to the main thread
                signals.report_ready.emit(report_msg)

            except Exception as e:
                signals.log.emit(f"❌ Sanity check error: {e}")
                signals.error.emit(str(e))

            finally:
                signals.finished.emit()

        threading.Thread(target=worker, daemon=True).start()


    # Helper method in your class to show the report safely in main thread
    def _show_sanity_report(self, msg):
        try:
            with open("sanity_check_report.txt", "w", encoding="utf-8") as f:
                f.write(msg)
        except Exception as e:
            self.log(f"⚠️ Could not save report: {e}")
        QMessageBox.information(self, "Sanity Check Result", msg)


    def laplacian_smoothing_gui(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        iterations, ok = QInputDialog.getInt(self, "Laplacian Smoothing", "Iterations:", 1, 1, 100, 1)
        if not ok:
            return
        lam, ok = QInputDialog.getDouble(self, "Laplacian Smoothing", "Lambda (0..1):", 0.5, 0.0, 1.0, 3)
        if not ok:
            return

        self.progress.show()
        self.progress.setValue(0)
        self.action_build.setEnabled(False)
        self.log("🛠️ Applying Laplacian smoothing...")

        signals = WorkerSignals()
        signals.progress.connect(self.progress.setValue)
        signals.finished.connect(lambda: self.progress.hide())
        signals.finished.connect(lambda: self.action_build.setEnabled(True))
        signals.log.connect(self.log)

        def worker():
            import time
            t0 = time.time()
            try:
                if self.state.get("original_vertices") is None:
                    self.state["original_vertices"] = copy.deepcopy(self.state["vertices"])

                vertices = self.state["vertices"]
                V = len(vertices)
                
                # Build adjacency list once
                adjacency = [[] for _ in range(V)]
                for edge in self.state["edges"]:
                    adjacency[edge.v1].append(edge.v2)
                    adjacency[edge.v2].append(edge.v1)

                coords = np.array([v.coords for v in vertices])
                original_coords = coords.copy()

                # Smoothing iterations with progress updates
                for it in range(iterations):
                    new_coords = coords.copy()
                    for i in range(V):
                        neighbors = adjacency[i]
                        if neighbors:
                            avg = coords[neighbors].mean(axis=0)
                            new_coords[i] = coords[i] + lam * (avg - coords[i])
                    coords = new_coords

                    # Emit progress (percentage)
                    pct = int((it + 1) / iterations * 100)
                    signals.progress.emit(pct)

                diff_vectors = coords - original_coords

                # Update vertices
                for i, v in enumerate(vertices):
                    v.coords = coords[i]
                self.state["vertices"] = vertices

                t1 = time.time()
                max_move = float(np.linalg.norm(diff_vectors, axis=1).max())
                signals.log.emit(f"✅ Laplacian smoothing done. Max move: {max_move:.4f}. Time: {t1 - t0:.3f}s")

                # Draw the updated mesh
                QTimer.singleShot(0, self.draw_mesh_from_vertices_triangles)

            except Exception as e:
                signals.log.emit(f"❌ Laplacian smoothing error: {e}")

            finally:
                signals.finished.emit()

        threading.Thread(target=worker, daemon=True).start()

    def taubin_smoothing_gui(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        iters, ok = QInputDialog.getInt(self, "Taubin Smoothing", "Iterations:", 10, 1, 100, 1)
        if not ok:
            return
        lam, ok = QInputDialog.getDouble(self, "Taubin Smoothing", "Lambda (positive):", 0.5, 0.0, 1.0, 3)
        if not ok:
            return
        mu, ok = QInputDialog.getDouble(self, "Taubin Smoothing", "Mu (negative):", -0.53, -5.0, 0.0, 3)
        if not ok:
            return

        self.progress.show()
        self.progress.setValue(0)
        self.action_build.setEnabled(False)
        self.log("🛠️ Applying Taubin smoothing...")

        signals = WorkerSignals()
        signals.progress.connect(self.progress.setValue)
        signals.finished.connect(lambda: self.progress.hide())
        signals.finished.connect(lambda: self.action_build.setEnabled(True))
        signals.log.connect(self.log)

        def worker():
            import time
            t0 = time.time()
            try:
                if self.state.get("original_vertices") is None:
                    self.state["original_vertices"] = copy.deepcopy(self.state["vertices"])

                vertices = self.state["vertices"]
                V = len(vertices)

                # Build adjacency list
                adjacency = [[] for _ in range(V)]
                for edge in self.state["edges"]:
                    adjacency[edge.v1].append(edge.v2)
                    adjacency[edge.v2].append(edge.v1)

                coords = np.array([v.coords for v in vertices])
                original_coords = coords.copy()

                def laplacian_step(coords, factor):
                    new_coords = coords.copy()
                    for i in range(V):
                        neighbors = adjacency[i]
                        if neighbors:
                            avg = coords[neighbors].mean(axis=0)
                            new_coords[i] += factor * (avg - coords[i])
                    return new_coords

                # Smoothing iterations with progress updates
                for it in range(iters):
                    coords = laplacian_step(coords, lam)
                    coords = laplacian_step(coords, mu)
                    pct = int((it + 1) / iters * 100)
                    signals.progress.emit(pct)

                diff_vectors = coords - original_coords

                # Update vertex objects
                for i, v in enumerate(vertices):
                    v.coords = coords[i]
                self.state["vertices"] = vertices

                t1 = time.time()
                max_move = float(np.linalg.norm(diff_vectors, axis=1).max())
                signals.log.emit(f"✅ Taubin smoothing done. Max move: {max_move:.4f}. Time: {t1 - t0:.3f}s")

                # Draw updated mesh
                QTimer.singleShot(0, self.draw_mesh_from_vertices_triangles)

            except Exception as e:
                signals.log.emit(f"❌ Taubin smoothing error: {e}")

            finally:
                signals.finished.emit()

        threading.Thread(target=worker, daemon=True).start()

    def compute_dihedral_angles(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        self.log("🧮 Computing dihedral angles...")

        def worker():
            try:
                MeshOperations.compute_triangle_normals(self.state["vertices"], self.state["triangles"])
                angles = compute_dihedral_angles(self.state["edges"], self.state["triangles"])
                vals = [a for a in angles.values() if a is not None]
                if not vals:
                    self.log("No non-boundary edges found.")
                    QTimer.singleShot(0, lambda: QMessageBox.information(self, "Dihedral Angles", "No non-boundary edges."))
                    return
                arr = np.array(vals)
                msg = (
                    f"Computed {len(arr)} edges.\n"
                    f"Min: {arr.min():.2f}°\nMax: {arr.max():.2f}°\n"
                    f"Mean: {arr.mean():.2f}°\nStd: {arr.std():.2f}°"
                )
                self.log("✅ Dihedral angles computed.")
                QTimer.singleShot(0, lambda: QMessageBox.information(self, "Dihedral Angles Statistics", msg))
            except Exception as e:
                self.log(f"❌ Dihedral error: {e}")
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def compute_point_mesh_distance_gui(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return

        s, ok = QInputDialog.getText(self, "Input Point", "x,y,z:")
        if not ok or not s:
            return
        try:
            x, y, z = [float(p.strip()) for p in s.split(",")]
            point = np.array([x, y, z], dtype=float)
        except Exception:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid 3D point as x,y,z.")
            return

        def worker():
            try:
                dist, closest = point_to_mesh_distance(point, self.state["vertices"], self.state["triangles"])
                self.log(f"✅ Point-mesh distance: {dist:.6f}. Closest: {closest}")
                QTimer.singleShot(0, lambda: QMessageBox.information(
                    self, "Point-Mesh Distance",
                    f"Shortest distance: {dist:.6f}\nClosest point on mesh: {closest}"
                ))
                # Visualize: add the two points as spheres
                def draw_points():
                    self.draw_mesh_from_vertices_triangles()
                    pts = np.array([vv.coords for vv in self.state["vertices"]])
                    r = 0.01 * np.linalg.norm(pts.max(0) - pts.min(0))
                    self.plotter.add_mesh(pv.Sphere(radius=r, center=point), color="red")
                    self.plotter.add_mesh(pv.Sphere(radius=r, center=closest), color="green")
                    self.plotter.reset_camera()
                QTimer.singleShot(0, draw_points)
            except Exception as e:
                self.log(f"❌ Distance error: {e}")
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def detect_and_smooth_cables(self):
        if not self.state["vertices"]:
            self.signals.show_message.emit("No Data", "Please build the structure first.")
            return

        # User inputs
        k_ring, ok = QInputDialog.getInt(self, "Detect Cables", "Neighborhood k_ring:", 8, 1, 10, 1)
        if not ok: return
        eig_ratio, ok = QInputDialog.getDouble(self, "Detect Cables", "Eigenvalue ratio threshold:", 0.25, 0.0, 1.0, 3)
        if not ok: return
        radius, ok = QInputDialog.getDouble(self, "Detect Cables", "Max radius threshold (0 = no limit):", 0.0, 0.0, 1e9, 6)
        if not ok: return
        if radius == 0.0: radius = None
        min_comp, ok = QInputDialog.getInt(self, "Detect Cables", "Min component size:", 30, 1, 100000, 1)
        if not ok: return

        self.signals.log.emit("🔎 Detecting cable-like regions...")

        def worker():
            try:
                mask, scores = detect_tubular_regions(
                    self.state["vertices"], self.state["edges"], self.state["triangles"],
                    k_ring=k_ring, eig_ratio_thresh=eig_ratio, radius_thresh=radius,
                    min_component_size=min_comp
                )
                num = int(mask.sum())
                self.signals.log.emit(f"Found {num} candidate cable vertices.")
                if num == 0:
                    self.signals.show_message.emit("Result", "No cable-like regions detected.")
                    return

                # Prepare preview colors in worker thread
                pts = np.array([v.coords for v in self.state["vertices"]])
                colors = np.zeros((len(self.state["vertices"]), 3))
                colors[mask] = np.array(pv.Color("red").int_rgb) / 255.0
                colors[~mask] = np.array(pv.Color("#ccf5ff").int_rgb) / 255.0

                # Emit signal to update mesh on main thread
                self.signals.mesh_update.emit(pts, colors)

                self.signals.log.emit("🧼 Smoothing detected cable regions...")
                vertices_new, diff = taubin_smoothing_masked(
                    self.state["vertices"], self.state["edges"], self.state["triangles"],
                    mask, iterations=30, lambda_factor=0.6, mu_factor=-0.62
                )
                self.state["vertices"] = vertices_new
                self.signals.log.emit("✅ Cable smoothing complete.")

                # Emit final mesh for main thread
                pts_final = np.array([v.coords for v in self.state["vertices"]])
                colors_final = np.zeros((len(self.state["vertices"]), 3))
                colors_final[mask] = np.array(pv.Color("red").int_rgb) / 255.0
                colors_final[~mask] = np.array(pv.Color("#ccf5ff").int_rgb) / 255.0
                self.signals.mesh_update.emit(pts_final, colors_final)

                self.signals.show_message.emit(
                    "Done", f"Detected {num} vertices in cable regions; smoothing applied."
                )

            except Exception as e:
                self.signals.error.emit(f"Cable detection/smoothing error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def qem_simplify_mesh(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return
        if QEM is None:
            QMessageBox.critical(self, "QEM Not Available", "QEM extension module not found.")
            return

        # Ask for percent reduction instead of target face count
        reduction_percent, ok = QInputDialog.getDouble(
            self, "QEM Simplification", "Reduction percent (0..100):",
            50.0, 1.0, 99.0, 1
        )
        if not ok:
            return

        self.signals.log.emit(f"🛠️ Simplifying mesh (QEM) by {reduction_percent:.1f}%...")
        self.progress.show()
        self.progress.setValue(0)
        self.action_qem.setEnabled(False)

        def worker():
            try:
                v_np, f_np = vertices_triangles_to_numpy(self.state["vertices"], self.state["triangles"])
                n_faces = f_np.shape[0]

                # Compute target faces based on percent reduction
                target_faces = max(1, int(n_faces * (1.0 - reduction_percent / 100.0)))

                # Python callback for progress
                def progress_cb(p):
                    self.signals.progress.emit(p)

                # Run C++ QEM simplify_mesh with callback
                new_v_np, new_f_np = QEM.simplify_mesh(v_np, f_np, target_faces, progress_cb)

                # Convert back to Vertex/Triangle objects
                new_vertices = [Vertex(coords=new_v_np[i], index=i) for i in range(len(new_v_np))]
                new_triangles = [Triangle(vertex_indices=list(new_f_np[i]), index=i) for i in range(len(new_f_np))]

                # Recompute normals
                for t in new_triangles:
                    t.recompute_normal(new_vertices)

                # --- Rebuild edges ---
                edge_dict = {}
                new_edges = []
                step = 0
                total_steps = len(new_triangles) * 3

                for tri in new_triangles:
                    vids = tri.vertex_indices
                    edge_vertices = [
                        (vids[1], vids[2]),
                        (vids[2], vids[0]),
                        (vids[0], vids[1]),
                    ]
                    for i, (v_start, v_end) in enumerate(edge_vertices):
                        key = tuple(sorted((v_start, v_end)))
                        if key in edge_dict:
                            edge_index = edge_dict[key]
                            new_edges[edge_index].triangles.append(tri.index)
                            tri.edge_indices[i] = edge_index
                        else:
                            edge_index = len(new_edges)
                            edge = Edge(v1=key[0], v2=key[1])
                            edge.triangles.append(tri.index)
                            new_edges.append(edge)
                            edge_dict[key] = edge_index
                            tri.edge_indices[i] = edge_index

                        step += 1
                        if step % 50 == 0:
                            progress_percent = int((step / total_steps) * 100)
                            self.signals.progress.emit(progress_percent)

                # --- Update state ---
                self.state["vertices"] = new_vertices
                self.state["triangles"] = new_triangles
                self.state["edges"] = new_edges

                self.signals.log.emit(f"✅ QEM complete. Faces: {len(new_triangles)}, Edges: {len(new_edges)}, Vertices: {len(new_vertices)}")

                # Update plot (white colors)
                pts, _ = vertices_triangles_to_numpy(new_vertices, new_triangles)
                colors = np.ones((len(new_vertices), 3))
                self.signals.mesh_update.emit(pts, colors)

                self.signals.progress.emit(100)

            except Exception as e:
                self.signals.log.emit(f"❌ QEM error: {e}")
                self.signals.show_message.emit("QEM Error", str(e))

            finally:
                self.action_qem.setEnabled(True)
                self.progress.hide()

        threading.Thread(target=worker, daemon=True).start()

def main():
    # Required for PyVista + Qt to cooperate nicely
    pv.set_plot_theme("dark")
    app = QApplication(sys.argv)
    w = MeshApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
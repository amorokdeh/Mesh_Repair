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
    QLabel,
    QDialog,
    QPushButton
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
    taubin_smoothing_masked,
    fill_mesh_holes
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

try:
    import curvature_simplification as CURV
except Exception:
    CURV = None
    print("Warning: curvature_simplification extension not found; curvature simplification unavailable.")


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

        self.action_fill = QAction("Fill Holes", self)
        self.action_fill.setEnabled(False)
        self.action_fill.triggered.connect(self.fill_holes_preview)
        actions_menu.addAction(self.action_fill)
        
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

        self.action_curvature = QAction("Simplify Mesh (Curvature)", self)
        self.action_curvature.setEnabled(False and (CURV is not None))
        self.action_curvature.triggered.connect(self.curvature_simplify_mesh)
        actions_menu.addAction(self.action_curvature)

        # --- App state ---
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
        self.action_curvature.setEnabled(False and (CURV is not None))


    def set_actions_enabled_for_built(self, enabled: bool):
        self.action_export.setEnabled(enabled)
        self.action_sanity.setEnabled(enabled)
        self.action_fill.setEnabled(enabled)
        self.action_lap.setEnabled(enabled)
        self.action_taubin.setEnabled(enabled)
        self.action_cables.setEnabled(enabled)
        self.action_dihedral.setEnabled(enabled)
        self.action_pointdist.setEnabled(enabled)
        self.action_qem.setEnabled(enabled and (QEM is not None))
        self.action_curvature.setEnabled(enabled and (CURV is not None))


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
            self, "Export Mesh", "", "STL (*.stl);;JSON (*.json)"
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
        if not hasattr(signals, "report_ready"):
            signals.report_ready = pyqtSignal(str)
        if not hasattr(signals, "mesh_update"):
            signals.mesh_update = pyqtSignal(object, object) 

        # Connect signals
        signals.progress.connect(self.progress.setValue)
        signals.finished.connect(lambda: self.progress.hide())
        signals.finished.connect(lambda: self.log("✅ Sanity check complete."))
        signals.log.connect(self.log)
        signals.report_ready.connect(self._show_sanity_report)  # will show the report
        signals.mesh_update.connect(self._preview_sanity_mesh)

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

                # Create colored mesh ---
                pts = np.array([v.coords for v in self.state["vertices"]])
                colors = np.tile(np.array([0.8, 0.95, 1.0]), (len(self.state["vertices"]), 1))
                if results["boundary_vertices"]:
                    colors[list(results["boundary_vertices"])] = np.array(pv.Color("red").int_rgb) / 255.0

                signals.mesh_update.emit(pts, colors)

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

        # --- Custom styled dialog instead of QMessageBox ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Sanity Check Result")
        dlg.resize(800, 600)  # make it large enough

        layout = QVBoxLayout(dlg)

        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setPlainText(msg)
        # Dark background + white text styling
        text_box.setStyleSheet("""
            QTextEdit {
                background-color: #121212;
                color: #ffffff;
                font-family: Consolas, monospace;
                font-size: 12pt;
                border: none;
            }
        """)

        # Add a Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        layout.addWidget(text_box)
        layout.addWidget(close_btn)

        dlg.exec_()


    def _preview_sanity_mesh(self, pts, colors):
        """Update mesh preview with sanity-check coloring (red = hole boundary)."""
        try:
            self.signals.mesh_update.emit(pts, colors)
        except Exception as e:
            self.log(f"⚠️ Could not update sanity preview: {e}")


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

    def fill_holes_preview(self):
        if not self.state["vertices"]:
            self.signals.show_message.emit("No Data", "Please build the structure first.")
            return

        self.signals.log.emit("🩹 Detecting and filling mesh holes...")

        def worker():
            try:
                vertices = self.state["vertices"]
                edges = self.state["edges"]
                triangles = self.state["triangles"]

                # Keep original triangle set
                original_tri_indices = set(t.index for t in triangles)

                # Fill holes
                vertices_new, edges_new, triangles_new = fill_mesh_holes(vertices, edges, triangles, max_loop_size=200)

                # Detect newly added triangles
                new_triangles = [t for t in triangles_new if t.index not in original_tri_indices]

                # Collect all vertices involved in new triangles
                filled_vertex_indices = set()
                for tri in new_triangles:
                    filled_vertex_indices.update(tri.vertex_indices)

                # --- Compact vertices to remove unused ---
                used_vertex_ids = np.unique([vi for tri in triangles_new for vi in tri.vertex_indices])
                old_to_new = -np.ones(len(vertices_new), dtype=int)
                for new_idx, old_idx in enumerate(used_vertex_ids):
                    old_to_new[old_idx] = new_idx

                # Rebuild vertex list
                compact_vertices = [vertices_new[i] for i in used_vertex_ids]

                # Remap triangle vertex indices
                for tri in triangles_new:
                    tri.vertex_indices = [old_to_new[vi] for vi in tri.vertex_indices]

                # Prepare colors
                pts = np.array([v.coords for v in compact_vertices])
                colors = np.tile(np.array([0.8, 0.95, 1.0]), (len(compact_vertices), 1))  # default color
                if filled_vertex_indices:
                    # map old indices to new compact indices
                    filled_idx_list = [old_to_new[vi] for vi in filled_vertex_indices if old_to_new[vi] >= 0]
                    colors[filled_idx_list] = np.array(pv.Color("red").int_rgb) / 255.0

                # Emit mesh preview to UI
                self.signals.mesh_update.emit(pts, colors)

                # Update app state
                self.state["vertices"] = compact_vertices
                self.state["edges"] = edges_new
                self.state["triangles"] = triangles_new

                self.signals.log.emit(f"✅ Hole filling complete. {len(filled_vertex_indices)} vertices in filled holes.")
                self.signals.show_message.emit(
                    "Done", f"Filled holes affecting {len(filled_vertex_indices)} vertices."
                )

            except Exception as e:
                self.signals.error.emit(f"Hole filling error: {e}")

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

        signals = WorkerSignals()
        if not hasattr(signals, "report_ready"):
            signals.report_ready = pyqtSignal(str)

        signals.report_ready.connect(self._show_dihedral_report)

        def worker():
            try:
                MeshOperations.compute_triangle_normals(self.state["vertices"], self.state["triangles"])
                angles = compute_dihedral_angles(self.state["edges"], self.state["triangles"])
                vals = [a for a in angles.values() if a is not None]

                if not vals:
                    signals.report_ready.emit("No non-boundary edges found.")
                    return

                arr = np.array(vals)
                msg = (
                    f"Computed {len(arr)} edges.\n"
                    f"Min: {arr.min():.2f}°\nMax: {arr.max():.2f}°\n"
                    f"Mean: {arr.mean():.2f}°\nStd: {arr.std():.2f}°"
                )
                self.log("✅ Dihedral angles computed.")
                signals.report_ready.emit(msg)

            except Exception as e:
                self.log(f"❌ Dihedral error: {e}")
                signals.report_ready.emit(f"Error computing dihedral angles:\n{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _show_dihedral_report(self, msg):
        # Console log also
        self.log(msg)

        # Dialog with same style as sanity check
        dlg = QDialog(self)
        dlg.setWindowTitle("Dihedral Angles Result")
        dlg.resize(500, 400)

        layout = QVBoxLayout(dlg)

        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setPlainText(msg)
        text_box.setStyleSheet("""
            QTextEdit {
                background-color: #121212;
                color: #ffffff;
                font-family: Consolas, monospace;
                font-size: 12pt;
                border: none;
            }
        """)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        layout.addWidget(text_box)
        layout.addWidget(close_btn)

        dlg.exec_()

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

        # Connect the update_preview signal to the UI handler only once
        # (update_preview is pyqtSignal(object) defined in WorkerSignals)
        if not hasattr(self, "_point_distance_connected") or not self._point_distance_connected:
            self.signals.update_preview.connect(self._show_point_mesh_distance)
            self._point_distance_connected = True

        def worker():
            try:
                dist, closest = point_to_mesh_distance(point, self.state["vertices"], self.state["triangles"])
                result = {"dist": dist, "closest": closest, "point": point}
                # Emit an object (dict) — update_preview accepts object types.
                self.signals.update_preview.emit(result)
            except Exception as e:
                # Send error back as object so main thread can show it
                self.signals.update_preview.emit({"error": str(e)})

        threading.Thread(target=worker, daemon=True).start()


    def _show_point_mesh_distance(self, result):
        """
        Runs in the main GUI thread (connected to self.signals.update_preview).
        Displays the result in a styled dialog and draws the two spheres on the plotter.
        """
        try:
            if not isinstance(result, dict):
                # safety fallback
                QMessageBox.critical(self, "Error", "Unexpected result type from worker.")
                return

            if "error" in result:
                QMessageBox.critical(self, "Error", result["error"])
                return

            dist = result["dist"]
            closest = result["closest"]
            point = result["point"]

            # Log to app log
            self.log(f"✅ Point-mesh distance: {dist:.6f}. Closest: {closest}")

            # Styled result dialog (consistent with your sanity dialog style)
            dlg = QDialog(self)
            dlg.setWindowTitle("Point-Mesh Distance Result")
            dlg.resize(520, 300)

            layout = QVBoxLayout(dlg)
            text_box = QTextEdit()
            text_box.setReadOnly(True)
            text_box.setPlainText(
                f"Shortest distance: {dist:.6f}\nClosest point on mesh: {closest}"
            )
            text_box.setStyleSheet("""
                QTextEdit {
                    background-color: #121212;
                    color: #ffffff;
                    font-family: Consolas, monospace;
                    font-size: 11pt;
                    border: none;
                }
            """)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)

            layout.addWidget(text_box)
            layout.addWidget(close_btn)
            dlg.exec_()

            # --- Visualization: draw spheres on the main plotter (main thread) ---
            # Redraw current mesh from vertices/triangles then add spheres on top
            try:
                self.draw_mesh_from_vertices_triangles()
            except Exception:
                # fallback: build a quick mesh if draw helper not available
                pts = np.array([v.coords for v in self.state["vertices"]])
                faces = []
                for t in self.state["triangles"]:
                    faces.extend([3] + list(t.vertex_indices))
                mesh = pv.PolyData(pts, np.array(faces))
                self.plotter.clear()
                self.plotter.add_mesh(mesh, color="#ccf5ff", show_edges=True, edge_color="#001f3f")
                self.plotter.reset_camera()

            # compute reasonable sphere radius
            pts_all = np.array([vv.coords for vv in self.state["vertices"]])
            bbox_size = np.linalg.norm(pts_all.max(axis=0) - pts_all.min(axis=0))
            r = max(1e-6, 0.01 * bbox_size)

            # Add sphere markers
            try:
                # Remove any previous marker actors with known names if you manage them,
                # else these will simply sit on top until next redraw.
                self.plotter.add_mesh(pv.Sphere(radius=r, center=point), color="red", name="query_point")
                self.plotter.add_mesh(pv.Sphere(radius=r, center=closest), color="green", name="closest_point")
                self.plotter.reset_camera()
            except Exception as e:
                # If plotter.add_mesh fails, log it but do not crash
                self.log(f"⚠️ Could not draw marker spheres: {e}")

        except Exception as e:
            # safety: show any unexpected errors
            self.log(f"❌ _show_point_mesh_distance error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def select_mesh_region(self, callback):
        """
        Let the user interactively select a region of the mesh using a draggable green box.
        Once 'Apply' is clicked, the selection bounds are passed to the callback(bounds).
        """
        import pyvista as pv
        from PyQt5.QtWidgets import QPushButton

        # Create the selection box widget
        bounds = self.plotter.bounds
        box_widget = [None]  # mutable holder for callback closure
        region_bounds = [None]

        def _callback(box):
            region_bounds[0] = box.bounds
            return

        # Add the box widget (no scaling_enabled argument!)
        box_widget[0] = self.plotter.add_box_widget(
            callback=_callback,
            bounds=bounds,
            color='green',
            outline_translation=True
        )

        self.log("🟩 Adjust the green box to select a region. Click 'Apply' when ready.")

        # --- Create floating Qt 'Apply' button ---
        btn_apply = QPushButton("Apply", self)
        btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        btn_apply.resize(100, 36)
        btn_apply.move(20, self.height() - 60)
        btn_apply.show()

        # --- Cleanup logic ---
        def cleanup_after_selection():
            try:
                if box_widget[0]:
                    self.plotter.remove_box_widget(box_widget[0])
            except Exception:
                pass
            btn_apply.deleteLater()

        # --- Apply selection ---
        def apply_selection():
            cleanup_after_selection()
            if region_bounds[0] is not None:
                callback(region_bounds[0])
            else:
                self.signals.log.emit("⚠️ No region selected.")

        btn_apply.clicked.connect(apply_selection)


    def detect_and_smooth_cables(self):
        if not self.state["vertices"]:
            self.signals.show_message.emit("No Data", "Please build the structure first.")
            return

        def on_region_selected(bounds):
            # Get user parameters after region selection
            k_ring, ok = QInputDialog.getInt(self, "Detect Cables", "Neighborhood k_ring:", 8, 1, 10, 1)
            if not ok: return
            eig_ratio, ok = QInputDialog.getDouble(self, "Detect Cables", "Eigenvalue ratio threshold:", 0.25, 0.0, 1.0, 3)
            if not ok: return
            radius, ok = QInputDialog.getDouble(self, "Detect Cables", "Max radius threshold (0 = no limit):", 0.0, 0.0, 1e9, 6)
            if not ok: return
            if radius == 0.0: radius = None
            min_comp, ok = QInputDialog.getInt(self, "Detect Cables", "Min component size:", 30, 1, 100000, 1)
            if not ok: return

            self.signals.log.emit("🔎 Detecting cable-like regions in selected area...")

            # Build vertex selection mask
            pts = np.array([v.coords for v in self.state["vertices"]])
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            region_mask = (
                (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
                (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
                (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
            )

            def worker():
                try:
                    mask, scores = detect_tubular_regions(
                        self.state["vertices"], self.state["edges"], self.state["triangles"],
                        k_ring=k_ring, eig_ratio_thresh=eig_ratio, radius_thresh=radius,
                        min_component_size=min_comp
                    )

                    # Restrict to region
                    mask = mask & region_mask
                    num = int(mask.sum())
                    self.signals.log.emit(f"Found {num} candidate cable vertices in region.")
                    if num == 0:
                        self.signals.show_message.emit("Result", "No cable-like regions detected in selection.")
                        return

                    # Visualization colors
                    pts = np.array([v.coords for v in self.state["vertices"]])
                    colors = np.zeros((len(pts), 3))
                    colors[mask] = np.array(pv.Color("red").int_rgb) / 255.0
                    colors[~mask] = np.array(pv.Color("#ccf5ff").int_rgb) / 255.0
                    self.signals.mesh_update.emit(pts, colors)

                    self.signals.log.emit("🧼 Smoothing detected cables in selected region...")
                    vertices_new, diff = taubin_smoothing_masked(
                        self.state["vertices"], self.state["edges"], self.state["triangles"],
                        mask, iterations=30, lambda_factor=0.6, mu_factor=-0.62
                    )
                    self.state["vertices"] = vertices_new
                    self.signals.log.emit("✅ Cable smoothing complete.")

                    pts_final = np.array([v.coords for v in self.state["vertices"]])
                    colors_final = np.zeros((len(pts_final), 3))
                    colors_final[mask] = np.array(pv.Color("red").int_rgb) / 255.0
                    colors_final[~mask] = np.array(pv.Color("#ccf5ff").int_rgb) / 255.0
                    self.signals.mesh_update.emit(pts_final, colors_final)

                    self.signals.show_message.emit(
                        "Done", f"Detected {num} vertices in cable regions within selected area; smoothing applied."
                    )

                except Exception as e:
                    self.signals.error.emit(f"Cable detection/smoothing error: {e}")

            threading.Thread(target=worker, daemon=True).start()

        # Start by letting user select region
        self.select_mesh_region(on_region_selected)


    def curvature_simplify_mesh(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return
        if CURV is None:
            QMessageBox.critical(self, "Curvature Not Available", "curvature_simplification extension not found.")
            return

        # Ask for percent reduction
        reduction_percent, ok = QInputDialog.getDouble(
            self, "Curvature Simplification",
            "Reduction percent (0..100):", 50.0, 1.0, 99.0, 1
        )
        if not ok:
            return

        self.signals.log.emit(f"🛠️ Simplifying mesh (Curvature) by {reduction_percent:.1f}%...")
        self.progress.show()
        self.progress.setValue(0)
        self.action_curvature.setEnabled(False)

        def worker():
            try:
                v_np, f_np = vertices_triangles_to_numpy(self.state["vertices"], self.state["triangles"])
                n_faces = f_np.shape[0]
                target_faces = max(1, int(n_faces * (1.0 - reduction_percent / 100.0)))

                # Progress callback
                def progress_cb(p):
                    self.signals.progress.emit(p)

                # Run C++ curvature simplification
                new_v_np, new_f_np = CURV.simplify_mesh_curvature(
                    v_np, f_np, target_faces,
                    1.0, 10.0, 2.0,  # alpha, beta, boundary_penalty
                    progress_cb
                )

                # Convert back to Vertex/Triangle
                new_vertices = [Vertex(coords=new_v_np[i], index=i) for i in range(len(new_v_np))]
                new_triangles = [Triangle(vertex_indices=list(new_f_np[i]), index=i) for i in range(len(new_f_np))]

                # Recompute normals
                for t in new_triangles:
                    t.recompute_normal(new_vertices)

                # Rebuild edges
                edge_dict = {}
                new_edges = []
                for tri in new_triangles:
                    vids = tri.vertex_indices
                    edge_vertices = [(vids[1], vids[2]), (vids[2], vids[0]), (vids[0], vids[1])]
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

                # Update state
                self.state["vertices"] = new_vertices
                self.state["triangles"] = new_triangles
                self.state["edges"] = new_edges

                self.signals.log.emit(f"✅ Curvature simplification complete. Faces: {len(new_triangles)}")

                # Update viewer
                pts, _ = vertices_triangles_to_numpy(new_vertices, new_triangles)
                colors = np.ones((len(new_vertices), 3))
                self.signals.mesh_update.emit(pts, colors)

                self.signals.progress.emit(100)

            except Exception as e:
                self.signals.log.emit(f"❌ Curvature simplification error: {e}")
                self.signals.show_message.emit("Curvature Error", str(e))

            finally:
                self.action_curvature.setEnabled(True)
                self.progress.hide()

        threading.Thread(target=worker, daemon=True).start()

    def qem_simplify_mesh(self):
        if not self.state["vertices"]:
            QMessageBox.warning(self, "No Data", "Please build the structure first.")
            return
        if QEM is None:
            QMessageBox.critical(self, "QEM Not Available", "QEM extension module not found.")
            return

        # Ask user for percent reduction
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
                # Convert current state to numpy arrays
                v_np, f_np = vertices_triangles_to_numpy(self.state["vertices"], self.state["triangles"])
                n_faces = f_np.shape[0]

                # Compute target faces
                target_faces = max(1, int(n_faces * (1.0 - reduction_percent / 100.0)))

                # Progress callback
                def progress_cb(p):
                    self.signals.progress.emit(max(0, min(100, int(p))))

                # --- QEM simplification ---
                new_v_np, new_f_np = QEM.simplify_mesh(v_np, f_np, target_faces, progress_cb)

                # --- FORCE CLEAN NUMERIC CONVERSION ---
                new_v = np.array([np.array(v, dtype=np.float64) for v in new_v_np], dtype=np.float64)
                new_f = np.array([np.array(f, dtype=np.int64) for f in new_f_np], dtype=np.int64)

                # --- Compact vertices to remove unused ---
                used_vertex_ids = np.unique(new_f.reshape(-1))
                old_to_compact = -np.ones(new_v.shape[0], dtype=np.int64)
                for compact_idx, old_idx in enumerate(used_vertex_ids):
                    old_to_compact[old_idx] = compact_idx

                if used_vertex_ids.size != new_v.shape[0]:
                    compact_v = new_v[used_vertex_ids]
                    remapped_f = np.array([[old_to_compact[idx] for idx in face] for face in new_f],
                                        dtype=np.int64)
                else:
                    compact_v = new_v
                    remapped_f = new_f.astype(np.int64)

                # --- Build Vertex and Triangle objects ---
                new_vertices = [Vertex(coords=np.array(compact_v[i], dtype=np.float64), index=i)
                                for i in range(compact_v.shape[0])]
                new_triangles = [Triangle(vertex_indices=[int(remapped_f[i, 0]),
                                                        int(remapped_f[i, 1]),
                                                        int(remapped_f[i, 2])],
                                        index=i)
                                for i in range(remapped_f.shape[0])]

                # --- Rebuild edges ---
                edge_dict = {}
                new_edges = []
                for tri in new_triangles:
                    vids = tri.vertex_indices
                    tri.edge_indices = []
                    edge_vertices = [(vids[0], vids[1]), (vids[1], vids[2]), (vids[2], vids[0])]
                    for v_start, v_end in edge_vertices:
                        key = (min(v_start, v_end), max(v_start, v_end))
                        if key in edge_dict:
                            edge_index = edge_dict[key]
                            new_edges[edge_index].triangles.append(tri.index)
                            tri.edge_indices.append(edge_index)
                        else:
                            e = Edge(v1=key[0], v2=key[1])
                            e.triangles.append(tri.index)
                            edge_index = len(new_edges)
                            new_edges.append(e)
                            edge_dict[key] = edge_index
                            tri.edge_indices.append(edge_index)

                # --- Compute normals and valence ---
                for v in new_vertices:
                    v.valence = 0
                    v.triangle_indices = []

                for tri in new_triangles:
                    p0, p1, p2 = [new_vertices[i].coords for i in tri.vertex_indices]
                    n = np.cross(p1 - p0, p2 - p0)
                    norm = np.linalg.norm(n)
                    tri.normal = n / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
                    for vi in tri.vertex_indices:
                        new_vertices[vi].valence += 1
                        new_vertices[vi].triangle_indices.append(tri.index)

                # --- Fix non-manifold edges ---
                def fix_non_manifold_edges(vertices, triangles, edges):
                    edges_to_remove = [e for e in edges if len(e.triangles) > 2]
                    bad_tri_indices = set()
                    for e in edges_to_remove:
                        bad_tri_indices.update(e.triangles)
                    triangles = [t for t in triangles if t.index not in bad_tri_indices]

                    # Rebuild edges
                    edge_dict = {}
                    new_edges = []
                    for t in triangles:
                        t.edge_indices = []
                        vids = t.vertex_indices
                        edge_vertices = [(vids[0], vids[1]), (vids[1], vids[2]), (vids[2], vids[0])]
                        for v_start, v_end in edge_vertices:
                            key = tuple(sorted((v_start, v_end)))
                            if key in edge_dict:
                                edge_index = edge_dict[key]
                                new_edges[edge_index].triangles.append(t.index)
                                t.edge_indices.append(edge_index)
                            else:
                                e = Edge(v1=key[0], v2=key[1])
                                e.triangles.append(t.index)
                                edge_index = len(new_edges)
                                new_edges.append(e)
                                edge_dict[key] = edge_index
                                t.edge_indices.append(edge_index)
                    return triangles, new_edges

                new_triangles, new_edges = fix_non_manifold_edges(new_vertices, new_triangles, new_edges)

                # --- Fill holes ---
                self.signals.log.emit("🩹 Detecting & filling holes...")
                new_vertices, new_edges, new_triangles = fill_mesh_holes(new_vertices, new_edges, new_triangles, max_loop_size=500)

                # --- Remove isolated vertices ---
                used_vertex_ids = set()
                for tri in new_triangles:
                    used_vertex_ids.update(tri.vertex_indices)

                old_to_new_idx = {}
                new_vertices_compact = []
                for v in new_vertices:
                    if v.index in used_vertex_ids:
                        old_to_new_idx[v.index] = len(new_vertices_compact)
                        new_vertices_compact.append(v)

                new_triangles_compact = []
                for tri in new_triangles:
                    tri.vertex_indices = [old_to_new_idx[i] for i in tri.vertex_indices]
                    new_triangles_compact.append(tri)

                new_vertices = new_vertices_compact
                new_triangles = new_triangles_compact

                # --- Rebuild edges after hole-filling and vertex compaction ---
                edge_dict = {}
                new_edges = []
                for tri in new_triangles:
                    tri.edge_indices = []
                    vids = tri.vertex_indices
                    for v_start, v_end in [(vids[0], vids[1]), (vids[1], vids[2]), (vids[2], vids[0])]:
                        key = (min(v_start, v_end), max(v_start, v_end))
                        if key in edge_dict:
                            edge_index = edge_dict[key]
                            new_edges[edge_index].triangles.append(tri.index)
                            tri.edge_indices.append(edge_index)
                        else:
                            e = Edge(v1=key[0], v2=key[1])
                            e.triangles.append(tri.index)
                            edge_index = len(new_edges)
                            new_edges.append(e)
                            edge_dict[key] = edge_index
                            tri.edge_indices.append(edge_index)

                # --- Recompute vertex normals ---
                MeshOperations.compute_vertex_normals(new_vertices, new_triangles)

                # --- Preserve colors ---
                if self.state.get("colors") is not None:
                    try:
                        old_colors = np.asarray(self.state["colors"])
                        if old_colors.shape[0] == v_np.shape[0]:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(v_np)
                            dists, idxs = tree.query(compact_v, k=1)
                            colors = old_colors[idxs]
                        else:
                            colors = np.tile(np.array([0.8, 0.95, 1.0]), (len(new_vertices), 1))
                    except Exception:
                        colors = np.tile(np.array([0.8, 0.95, 1.0]), (len(new_vertices), 1))
                else:
                    colors = np.tile(np.array([0.8, 0.95, 1.0]), (len(new_vertices), 1))

                # --- Update state ---
                self.state["vertices"] = new_vertices
                self.state["triangles"] = new_triangles
                self.state["edges"] = new_edges

                # --- Send updates to UI ---
                pts = np.array([v.coords for v in new_vertices], dtype=np.float64)
                self.signals.mesh_update.emit(pts, colors)
                self.signals.log.emit(
                    f"✅ QEM complete. Faces: {len(new_triangles)}, "
                    f"Edges: {len(new_edges)}, Vertices: {len(new_vertices)}"
                )
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
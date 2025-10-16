// curvature_simplification.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <functional>
#include <cmath>

namespace py = pybind11;

// -------------------- Utilities / Types --------------------
struct EdgeEntry {
    int v1, v2;               // vertices (v1 < v2)
    double cost;
    Eigen::Vector3d pos;      // collapsed position (if used)
    int stamp;                // version / stamp for staleness
    EdgeEntry(int a, int b, double c, const Eigen::Vector3d& p, int s)
        : v1(a), v2(b), cost(c), pos(p), stamp(s) {}
};

// comparator for min-heap
struct EdgeCompare {
    bool operator()(EdgeEntry const& a, EdgeEntry const& b) const {
        return a.cost > b.cost;
    }
};

// -------------------- Geometry helpers --------------------
static inline Eigen::Vector3d face_normal(const Eigen::RowVector3d &p0,
                                          const Eigen::RowVector3d &p1,
                                          const Eigen::RowVector3d &p2) {
    Eigen::Vector3d e1 = (p1 - p0).transpose();
    Eigen::Vector3d e2 = (p2 - p0).transpose();
    Eigen::Vector3d n = e1.cross(e2);
    double len = n.norm();
    if (len > 0) n /= len;
    return n;
}

static inline std::pair<int,int> make_edge_key(int a, int b) {
    if (a < b) return {a,b};
    return {b,a};
}

// -------------------- Core simplifier --------------------
py::tuple simplify_mesh_curvature(
    py::array_t<double> vertices_np,
    py::array_t<int> faces_np,
    int target_faces,
    double alpha = 1.0,
    double beta = 10.0,
    double boundary_penalty = 2.0,
    py::function progress_callback = py::none()
) {
    // --- Buffer access ---
    auto vbuf = vertices_np.request();
    auto fbuf = faces_np.request();
    const int nV = (int)vbuf.shape[0];
    const int nF = (int)fbuf.shape[0];

    Eigen::MatrixXd V(nV, 3);
    Eigen::MatrixXi F(nF, 3);

    double *vptr = (double*)vbuf.ptr;
    int *fptr = (int*)fbuf.ptr;
    for (int i = 0; i < nV; ++i) {
        V(i,0) = vptr[i*3 + 0];
        V(i,1) = vptr[i*3 + 1];
        V(i,2) = vptr[i*3 + 2];
    }
    for (int i = 0; i < nF; ++i) {
        F(i,0) = fptr[i*3 + 0];
        F(i,1) = fptr[i*3 + 1];
        F(i,2) = fptr[i*3 + 2];
    }

    if (target_faces < 1) target_faces = 1;
    if (target_faces >= nF) {
        // nothing to do, return input
        py::array_t<double> out_v({nV,3});
        py::array_t<int> out_f({nF,3});
        auto outv = out_v.mutable_unchecked<2>();
        auto outf = out_f.mutable_unchecked<2>();
        for (int i=0;i<nV;i++){
            outv(i,0)=V(i,0); outv(i,1)=V(i,1); outv(i,2)=V(i,2);
        }
        for (int i=0;i<nF;i++){
            outf(i,0)=F(i,0); outf(i,1)=F(i,1); outf(i,2)=F(i,2);
        }
        return py::make_tuple(out_v, out_f);
    }

    // --- 1) Build face normals ---
    std::vector<Eigen::Vector3d> face_normals(nF);
    for (int i=0;i<nF;i++){
        face_normals[i] = face_normal(V.row(F(i,0)), V.row(F(i,1)), V.row(F(i,2)));
    }

    // --- 2) Build adjacency: vertex -> incident faces, vertex -> neighbors ---
    std::vector<std::vector<int>> v2faces(nV);
    std::vector<std::unordered_set<int>> v_neighbors(nV);
    for (int fi=0; fi<nF; ++fi) {
        for (int k=0;k<3;k++){
            int vi = F(fi,k);
            v2faces[vi].push_back(fi);
            int vj = F(fi,(k+1)%3);
            int vk = F(fi,(k+2)%3);
            v_neighbors[vi].insert(vj);
            v_neighbors[vi].insert(vk);
        }
    }

    // --- 3) Vertex normals by averaging adjacent face normals ---
    std::vector<Eigen::Vector3d> vnorm(nV, Eigen::Vector3d::Zero());
    for (int vi=0; vi<nV; ++vi) {
        if (v2faces[vi].empty()) continue;
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();
        for (int fi : v2faces[vi]) acc += face_normals[fi];
        double len = acc.norm();
        if (len > 0) acc /= len;
        vnorm[vi] = acc;
    }

    // --- 4) Vertex curvature estimate: 1 - average dot(neighbor normals) ---
    std::vector<double> vcurv(nV, 0.0);
    for (int vi=0; vi<nV; ++vi) {
        const auto &nbrs = v_neighbors[vi];
        if (nbrs.empty()) { vcurv[vi] = 0.0; continue; }
        double sumdot = 0.0;
        for (int nb : nbrs) sumdot += std::max(-1.0, std::min(1.0, vnorm[vi].dot(vnorm[nb])));
        vcurv[vi] = 1.0 - (sumdot / double(nbrs.size()));
        if (vcurv[vi] < 0) vcurv[vi] = 0;
    }

    // --- 5) Build edge -> incident faces map ---
    using EdgeKey = std::pair<int,int>;
    struct EdgeKeyHash { size_t operator()(EdgeKey const& e) const noexcept {
        return (size_t(e.first) << 32) ^ (size_t(e.second));
    }};
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge2faces;
    edge2faces.reserve(nF * 3);
    for (int fi=0; fi<nF; ++fi) {
        int a = F(fi,0), b = F(fi,1), c = F(fi,2);
        EdgeKey e0 = make_edge_key(std::min(a,b), std::max(a,b));
        EdgeKey e1 = make_edge_key(std::min(b,c), std::max(b,c));
        EdgeKey e2 = make_edge_key(std::min(c,a), std::max(c,a));
        edge2faces[e0].push_back(fi);
        edge2faces[e1].push_back(fi);
        edge2faces[e2].push_back(fi);
    }

    // --- 6) Build initial edge heap with cost ---
    std::priority_queue<EdgeEntry, std::vector<EdgeEntry>, EdgeCompare> heap;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_stamp_map;
    int global_stamp = 1;

    auto edge_length = [&](int i, int j)->double {
        return (V.row(i) - V.row(j)).norm();
    };

    auto compute_edge_cost = [&](int i, int j)->std::pair<double, Eigen::Vector3d> {
        double L = edge_length(i,j);
        double cur = vcurv[i] + vcurv[j];
        double cost = alpha * L + beta * cur;
        EdgeKey ek = make_edge_key(std::min(i,j), std::max(i,j));
        auto it = edge2faces.find(ek);
        if (it == edge2faces.end() || it->second.size() != 2) cost += boundary_penalty;
        Eigen::Vector3d pos = 0.5 * (V.row(i).transpose() + V.row(j).transpose()); // midpoint
        return {cost, pos};
    };

    for (auto &kv : edge2faces) {
        int i = kv.first.first, j = kv.first.second;
        auto pr = compute_edge_cost(i,j);
        int st = global_stamp++;
        heap.emplace(i,j, pr.first, pr.second, st);
        edge_stamp_map[kv.first] = st;
    }

    // --- Bookkeeping for removals ---
    std::vector<char> removed_vertex(nV, 0);
    std::vector<char> removed_face(nF, 0);

    int current_faces = nF;
    int total_to_remove = std::max(0, nF - target_faces);
    int removed_faces_count = 0;
    int last_percent = -1;

    // Helper: replace vertex id 'oldv' -> 'keep' in face fi
    auto replace_vertex_in_face = [&](int fi, int oldv, int keep) {
        for (int k=0;k<3;k++){
            if (F(fi,k) == oldv) F(fi,k) = keep;
        }
    };

    // Helper: mark face degenerate (if duplicate vertex indices)
    auto is_degenerate_face = [&](int fi)->bool {
        int a = F(fi,0), b = F(fi,1), c = F(fi,2);
        return (a==b) || (b==c) || (c==a);
    };

    // Build face -> incident edges set (for local updates)
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge2faces_local = edge2faces; // copy

    // --- 7) Main collapse loop ---
    while (current_faces > target_faces && !heap.empty()) {
        EdgeEntry top = heap.top(); heap.pop();
        EdgeKey ek = make_edge_key(std::min(top.v1, top.v2), std::max(top.v1, top.v2));

        // stale check
        auto itstamp = edge_stamp_map.find(ek);
        if (itstamp == edge_stamp_map.end() || itstamp->second != top.stamp) {
            continue; // stale
        }

        int vkeep = top.v1;
        int vrem = top.v2;
        if (removed_vertex[vkeep] || removed_vertex[vrem]) continue;

        // Recompute the cost (to be safe) and re-check boundary status
        auto recomputed = compute_edge_cost(vkeep, vrem);
        double recomputed_cost = recomputed.first;
        // allow small tolerance: if recomputed cost is significantly different, push updated entry
        if (std::abs(recomputed_cost - top.cost) > 1e-9) {
            int st = global_stamp++;
            heap.emplace(vkeep, vrem, recomputed_cost, recomputed.second, st);
            edge_stamp_map[ek] = st;
            continue;
        }

        // Apply collapse: move keep to new position
        Eigen::Vector3d newpos = top.pos;
        V.row(vkeep) = newpos.transpose();
        // mark removed vertex
        removed_vertex[vrem] = 1;

        // Collect incident faces that mention vrem or vkeep
        std::vector<int> incident_faces;
        incident_faces.reserve(32);
        for (int fidx : v2faces[vrem]) incident_faces.push_back(fidx);
        for (int fidx : v2faces[vkeep]) incident_faces.push_back(fidx);

        // Remove duplicates
        std::sort(incident_faces.begin(), incident_faces.end());
        incident_faces.erase(std::unique(incident_faces.begin(), incident_faces.end()), incident_faces.end());

        // Update faces: replace vrem->vkeep, detect degenerates
        for (int fi : incident_faces) {
            if (removed_face[fi]) continue;
            replace_vertex_in_face(fi, vrem, vkeep);
            if (is_degenerate_face(fi)) {
                removed_face[fi] = 1;
                current_faces--;
                removed_faces_count++;
            } else {
                // update stored face normal
                face_normals[fi] = face_normal(V.row(F(fi,0)), V.row(F(fi,1)), V.row(F(fi,2)));
            }
        }

        // Update vertex -> faces adjacency for vkeep
        std::vector<int> new_v2faces;
        new_v2faces.reserve(v2faces[vkeep].size() + v2faces[vrem].size());
        for (int fi : v2faces[vkeep]) if (!removed_face[fi]) new_v2faces.push_back(fi);
        for (int fi : v2faces[vrem]) if (!removed_face[fi]) new_v2faces.push_back(fi);
        std::sort(new_v2faces.begin(), new_v2faces.end());
        new_v2faces.erase(std::unique(new_v2faces.begin(), new_v2faces.end()), new_v2faces.end());
        v2faces[vkeep] = std::move(new_v2faces);
        v2faces[vrem].clear();

        // Update neighbors sets for vkeep (merge neighbor lists)
        for (int nb : v_neighbors[vrem]) {
            if (nb == vkeep) continue;
            v_neighbors[vkeep].insert(nb);
            v_neighbors[nb].erase(vrem);
            v_neighbors[nb].insert(vkeep);
        }
        v_neighbors[vrem].clear();

        // Recompute vnorm and curvature for affected vertices (vkeep and its neighbors)
        std::vector<int> affected_vs;
        affected_vs.push_back(vkeep);
        for (int nb : v_neighbors[vkeep]) affected_vs.push_back(nb);
        for (int vi : affected_vs) {
            // recompute vertex normal
            Eigen::Vector3d acc = Eigen::Vector3d::Zero();
            for (int fi : v2faces[vi]) acc += face_normals[fi];
            double len = acc.norm();
            if (len > 0) acc /= len;
            vnorm[vi] = acc;
        }
        for (int vi : affected_vs) {
            if (v_neighbors[vi].empty()) { vcurv[vi] = 0.0; continue; }
            double sumdot = 0.0;
            for (int nb : v_neighbors[vi]) sumdot += std::max(-1.0, std::min(1.0, vnorm[vi].dot(vnorm[nb])));
            vcurv[vi] = 1.0 - (sumdot / double(v_neighbors[vi].size()));
            if (vcurv[vi] < 0) vcurv[vi] = 0.0;
        }

        // Update edge->faces map locally: rebuild edges around affected faces
        // For reliability we will rebuild the edges incident to affected faces
        // Remove the old entries for edges that referenced removed faces
        for (int fi : incident_faces) {
            // skip faces already removed
            if (removed_face[fi]) {
                // remove fi from all edge2faces lists
                int a = F(fi,0), b = F(fi,1), c = F(fi,2);
                // but since we changed F for degenerates, safe to continue
                continue;
            }
        }

        // Rebuild edge2faces for neighborhood (naive local rebuild)
        // We'll remove any edges that now have no faces and update costs for edges incident to vkeep and its neighbors
        std::vector<std::pair<int,int>> edges_to_update;
        std::unordered_set<std::pair<int,int>, EdgeKeyHash> touched_edges;

        // collect edges around vkeep
        for (int fi : v2faces[vkeep]) {
            int a = F(fi,0), b = F(fi,1), c = F(fi,2);
            std::pair<int,int> e0 = make_edge_key(a,b);
            std::pair<int,int> e1 = make_edge_key(b,c);
            std::pair<int,int> e2 = make_edge_key(c,a);
            touched_edges.insert(e0); touched_edges.insert(e1); touched_edges.insert(e2);
        }
        for (auto &e : touched_edges) edges_to_update.push_back(e);

        // For each touched edge, recompute cost and push to heap with new stamp
        for (auto &e : edges_to_update) {
            int a = e.first, b = e.second;
            if (removed_vertex[a] || removed_vertex[b]) continue;
            auto pr = compute_edge_cost(a,b);
            int st = global_stamp++;
            heap.emplace(a,b, pr.first, pr.second, st);
            edge_stamp_map[e] = st;
        }

        // Progress callback
        if (!progress_callback.is_none() && total_to_remove > 0) {
            int percent = int(100.0 * double(removed_faces_count) / double(total_to_remove));
            if (percent != last_percent) {
                try {
                    py::gil_scoped_acquire acquire;
                    progress_callback(percent);
                } catch (...) {}
                last_percent = percent;
            }
        }
    } // end while

    // --- 8) Pack remaining vertices/faces into output arrays ---
    // Build mapping old -> new indices for alive vertices
    std::vector<int> old2new(nV, -1);
    int nv_out = 0;
    for (int i=0;i<nV;i++){
        if (!removed_vertex[i]) old2new[i] = nv_out++;
    }

    int nf_out = 0;
    for (int fi=0; fi<nF; ++fi) if (!removed_face[fi]) nf_out++;

    py::array_t<double> out_v({nv_out, 3});
    py::array_t<int> out_f({nf_out, 3});
    auto outv = out_v.mutable_unchecked<2>();
    auto outf = out_f.mutable_unchecked<2>();

    // Fill vertices
    int idx = 0;
    for (int i=0;i<nV;i++){
        if (removed_vertex[i]) continue;
        outv(idx,0) = V(i,0);
        outv(idx,1) = V(i,1);
        outv(idx,2) = V(i,2);
        ++idx;
    }

    // Fill faces (remap indices)
    idx = 0;
    for (int fi=0; fi<nF; ++fi) {
        if (removed_face[fi]) continue;
        int a = F(fi,0), b = F(fi,1), c = F(fi,2);
        // faces must be non-degenerate
        outf(idx,0) = old2new[a];
        outf(idx,1) = old2new[b];
        outf(idx,2) = old2new[c];
        ++idx;
    }

    // final progress
    if (!progress_callback.is_none()) {
        try { py::gil_scoped_acquire acquire; progress_callback(100); } catch(...) {}
    }

    return py::make_tuple(out_v, out_f);
}

// -------------------- pybind11 module --------------------
PYBIND11_MODULE(curvature_simplification, m) {
    m.def("simplify_mesh_curvature",
        &simplify_mesh_curvature,
        py::arg("vertices_np"),
        py::arg("faces_np"),
        py::arg("target_faces"),
        py::arg("alpha") = 1.0,
        py::arg("beta") = 10.0,
        py::arg("boundary_penalty") = 2.0,
        py::arg("progress_callback") = py::none(),
        "Simplify mesh using curvature-weighted edge collapse.\n\n"
        "Parameters:\n"
        "  vertices_np: (N,3) float64 array\n"
        "  faces_np: (M,3) int32 array\n"
        "  target_faces: desired face count\n"
        "  alpha, beta, boundary_penalty: cost weights\n        ");
}
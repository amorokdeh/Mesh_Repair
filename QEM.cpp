#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <set>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <pybind11/functional.h>

namespace py = pybind11;

// ==================== QEM Utilities ====================
Eigen::Matrix4d compute_plane_quadric(const Eigen::Vector3d &v0,
                                      const Eigen::Vector3d &v1,
                                      const Eigen::Vector3d &v2)
{
    Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
    double len = n.norm();
    if (len != 0) n /= len;
    double d = -n.dot(v0);
    Eigen::Vector4d plane;
    plane << n, d;
    return plane * plane.transpose();
}

Eigen::Vector3d compute_optimal_position(const Eigen::Matrix4d &Q)
{
    Eigen::Matrix3d A = Q.block<3,3>(0,0);
    Eigen::Vector3d b = -Q.block<3,1>(0,3);
    if(A.determinant() > 1e-12)
        return A.colPivHouseholderQr().solve(b);
    else
        return Eigen::Vector3d::Zero(); // fallback
}

// ==================== Edge Structure ====================
struct Edge {
    int v1, v2;
    double cost;
    Eigen::Vector3d pos;

    Edge(int a, int b, double c, Eigen::Vector3d p) : v1(a), v2(b), cost(c), pos(p) {}
};

struct CompareEdge {
    bool operator()(const Edge &e1, const Edge &e2) {
        return e1.cost > e2.cost; // min-heap
    }
};

// ==================== Simplify Mesh ====================
py::tuple simplify_mesh(py::array_t<double> vertices_np,
                        py::array_t<int> faces_np,
                        int target_faces,
                        std::function<void(int)> progress_callback = nullptr)
{
    auto v_buf = vertices_np.request();
    auto f_buf = faces_np.request();
    int n_vertices = v_buf.shape[0];
    int n_faces = f_buf.shape[0];

    Eigen::MatrixXd vertices(n_vertices,3);
    Eigen::MatrixXi faces(n_faces,3);

    double* v_ptr = (double*)v_buf.ptr;
    int* f_ptr = (int*)f_buf.ptr;

    for(int i=0;i<n_vertices;i++){
        vertices.row(i) << v_ptr[i*3], v_ptr[i*3+1], v_ptr[i*3+2];
    }
    for(int i=0;i<n_faces;i++){
        faces.row(i) << f_ptr[i*3], f_ptr[i*3+1], f_ptr[i*3+2];
    }

    // 1. Compute per-vertex QEM
    std::vector<Eigen::Matrix4d> qem(n_vertices, Eigen::Matrix4d::Zero());
    for(int i=0;i<n_faces;i++){
        int a = faces(i,0), b = faces(i,1), c = faces(i,2);
        Eigen::Matrix4d K = compute_plane_quadric(vertices.row(a), vertices.row(b), vertices.row(c));
        qem[a] += K; qem[b] += K; qem[c] += K;
    }

    // 2. Build edges set
    std::set<std::pair<int,int>> edges_set;
    for(int i=0;i<n_faces;i++){
        int a = faces(i,0), b = faces(i,1), c = faces(i,2);
        edges_set.insert({std::min(a,b), std::max(a,b)});
        edges_set.insert({std::min(b,c), std::max(b,c)});
        edges_set.insert({std::min(c,a), std::max(c,a)});
    }

    // 3. Compute edge collapse cost
    std::priority_queue<Edge, std::vector<Edge>, CompareEdge> heap;
    for(auto &e : edges_set){
        int v1 = e.first, v2 = e.second;
        Eigen::Matrix4d Qsum = qem[v1] + qem[v2];
        Eigen::Vector3d pos = compute_optimal_position(Qsum);
        Eigen::Vector4d v4; v4 << pos,1.0;
        double cost = v4.transpose() * Qsum * v4;
        heap.emplace(v1,v2,cost,pos);
    }

    // 4. Collapse edges until target faces
    std::vector<bool> removed_vertex(n_vertices,false);
    std::vector<bool> removed_face(n_faces,false);

    int current_faces = n_faces;
    int total_faces_to_collapse = n_faces - target_faces;
    int last_percent = -1;

    while(current_faces > target_faces && !heap.empty()){
        Edge e = heap.top(); heap.pop();
        if(removed_vertex[e.v1] || removed_vertex[e.v2]) continue;

        int keep = e.v1;
        int remove = e.v2;
        vertices.row(keep) = e.pos;
        qem[keep] += qem[remove];
        removed_vertex[remove] = true;

        for(int i=0;i<n_faces;i++){
            if(removed_face[i]) continue;
            for(int j=0;j<3;j++){
                if(faces(i,j) == remove) faces(i,j) = keep;
            }
            if(faces(i,0)==faces(i,1) || faces(i,1)==faces(i,2) || faces(i,2)==faces(i,0)){
                removed_face[i] = true;
                current_faces--;
            }
        }

        if(progress_callback){
            int percent = int(100.0 * (total_faces_to_collapse - (current_faces - target_faces)) / total_faces_to_collapse);
            if(percent != last_percent){
                progress_callback(percent);
                last_percent = percent;
            }
        }
    }

    // 5. Collect remaining faces and vertices
    std::unordered_map<int,int> v_map;
    int idx=0;
    for(int i=0;i<n_vertices;i++){
        if(!removed_vertex[i]){
            v_map[i] = idx++;
        }
    }
    int n_new_vertices = v_map.size();
    int n_new_faces = 0;
    for(int i=0;i<n_faces;i++){
        if(!removed_face[i]) n_new_faces++;
    }

    py::array_t<double> out_v({n_new_vertices,3});
    py::array_t<int> out_f({n_new_faces,3});
    auto v_out = out_v.mutable_unchecked<2>();
    auto f_out = out_f.mutable_unchecked<2>();

    idx=0;
    std::vector<int> new_v_indices(n_vertices,-1);
    for(int i=0;i<n_vertices;i++){
        if(!removed_vertex[i]){
            v_out(idx,0) = vertices(i,0);
            v_out(idx,1) = vertices(i,1);
            v_out(idx,2) = vertices(i,2);
            new_v_indices[i] = idx;
            idx++;
        }
    }

    idx=0;
    for(int i=0;i<n_faces;i++){
        if(!removed_face[i]){
            f_out(idx,0) = new_v_indices[faces(i,0)];
            f_out(idx,1) = new_v_indices[faces(i,1)];
            f_out(idx,2) = new_v_indices[faces(i,2)];
            idx++;
        }
    }

    return py::make_tuple(out_v,out_f);
}

PYBIND11_MODULE(QEM, m) {
    m.def("simplify_mesh",
      [](py::array_t<double> vertices_np,
         py::array_t<int> faces_np,
         int target_faces,
         py::function progress_callback = py::none())
      {
          std::function<void(int)> cb = nullptr;
          if (!progress_callback.is_none()) {
              cb = [progress_callback](int p){
                  py::gil_scoped_acquire acquire;
                  progress_callback(p);
              };
          }
          return simplify_mesh(vertices_np, faces_np, target_faces, cb);
      },
      py::arg("vertices_np"),
      py::arg("faces_np"),
      py::arg("target_faces"),
      py::arg("progress_callback") = py::none(),
      "Simplify mesh using QEM (C++ accelerated) with optional progress callback");
}
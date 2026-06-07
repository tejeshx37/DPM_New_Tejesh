#include <unordered_map>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/mark_domain_in_triangulation.h>

#include "triangulation.h"

namespace Triangulation
{
    typedef CGAL::Triangulation_vertex_base_2<Kernel> Vb;
    typedef CGAL::Delaunay_mesh_face_base_2<Kernel> Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds> CDT;
    typedef CDT::Vertex_handle VertexHandle;
    typedef CDT::Face_handle FaceHandle;
    typedef CDT::Face_circulator FaceCirculator;
    typedef CDT::Edge Edge;
    typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
    typedef CGAL::Delaunay_mesher_2<CDT, Criteria> Mesher;

    typedef CGAL::Cartesian_converter<AlgebraicKernel, Kernel> CartesianConverter;

    std::vector<Face> &Data::faces() noexcept
    {
        return m_faces;
    }

    const std::vector<Face> &Data::faces() const noexcept
    {
        return m_faces;
    }

    std::vector<IndexPair> &Data::edges() noexcept
    {
        return m_edges;
    }

    const std::vector<IndexPair> &Data::edges() const noexcept
    {
        return m_edges;
    }

    std::vector<Vertex> &Data::vertices() noexcept
    {
        return m_vertices;
    }

    const std::vector<Vertex> &Data::vertices() const noexcept
    {
        return m_vertices;
    }

    std::unique_ptr<EpickPoint> create_epick_point(const double x, const double y)
    {
        return std::make_unique<EpickPoint>(x, y);
    }

    std::unique_ptr<PointPair> create_point_pair(const EpickPoint &first,
                                                 const EpickPoint &second)
    {
        return std::make_unique<PointPair>(first, second);
    }

    std::unique_ptr<Data> triangulate(const Constraints &constraints,
                                      const double aspect_bound,
                                      const double size_bound)
    {
        CDT cdt;
        for (const PointPair &point_pair : constraints)
        {
            cdt.insert_constraint(point_pair.first, point_pair.second);
        }

        CGAL::mark_domain_in_triangulation(cdt);

        Mesher mesher(cdt);
        mesher.init(true);
        mesher.set_criteria(Criteria(aspect_bound, size_bound));
        mesher.refine_mesh();

        auto data = std::make_unique<Data>();

        std::unordered_map<FaceHandle, std::size_t> face_index_map;
        face_index_map.reserve(cdt.number_of_faces());
        {
            std::size_t index = 0;
            for (const FaceHandle handle : cdt.finite_face_handles())
            {
                if (!handle->is_in_domain())
                {
                    continue;
                }
                face_index_map.emplace(handle, index++);
            }
        }

        std::unordered_map<VertexHandle, std::size_t> vertex_index_map;
        data->vertices().reserve(cdt.number_of_vertices());
        vertex_index_map.reserve(cdt.number_of_vertices());
        {
            std::size_t index = 0;
            for (const VertexHandle handle : cdt.finite_vertex_handles())
            {
                vertex_index_map.emplace(handle, index++);
                FaceCirculator face_circulator = handle->incident_faces(), done(face_circulator);
                std::vector<std::size_t> incident_faces;
                do
                {
                    const FaceHandle incident_face = face_circulator.base();
                    if (!incident_face->is_in_domain())
                    {
                        continue;
                    }
                    incident_faces.push_back(face_index_map.at(incident_face));
                } while (++face_circulator != done);
                data->vertices().emplace_back(handle->point(), std::move(incident_faces));
            }
        }

        for (const Edge &edge : cdt.finite_edges())
        {
            const FaceHandle f = edge.first;
            if (!f->is_in_domain()) {
                continue;
            }
            const auto vertex_index = [&](const int e)
            {
                return vertex_index_map.at(f->vertex(e));
            };
            data->edges().emplace_back(vertex_index(f->cw(edge.second)),
                                       vertex_index(f->ccw(edge.second)));
        }

        data->faces().reserve(cdt.number_of_faces());
        for (const FaceHandle handle : cdt.finite_face_handles())
        {
            if (!handle->is_in_domain())
            {
                continue;
            }
            const auto point_index = [&](const int i)
            {
                return vertex_index_map.at(handle->vertex(i));
            };
            data->faces()
                .push_back({point_index(0), point_index(1), point_index(2)});
        }

        return data;
    }
}
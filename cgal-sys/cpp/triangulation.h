#pragma once

#include <array>

#include "point.h"

namespace Triangulation
{
    using Kernel = EpicKernel;
    using EpickPoint = Kernel::Point_2;
    using PointPair = std::pair<EpickPoint, EpickPoint>;
    using Constraints = std::vector<PointPair>;
    using IndexPair = std::pair<std::size_t, std::size_t>;
    using Face = std::array<std::size_t, 3>;
    using Vertex = std::pair<EpickPoint, std::vector<std::size_t>>;

    class Data
    {
    public:
        std::vector<Face> &faces() noexcept;
        const std::vector<Face> &faces() const noexcept;

        std::vector<IndexPair> &edges() noexcept;
        const std::vector<IndexPair> &edges() const noexcept;

        std::vector<Vertex> &vertices() noexcept;
        const std::vector<Vertex> &vertices() const noexcept;

    private:
        std::vector<Face> m_faces;
        std::vector<IndexPair> m_edges;
        std::vector<Vertex> m_vertices;
    };

    std::unique_ptr<EpickPoint> create_epick_point(const double x, const double y);

    std::unique_ptr<PointPair> create_point_pair(const EpickPoint &first,
                                                 const EpickPoint &second);

    std::unique_ptr<Data> triangulate(const Constraints &constraints,
                                      const double aspect_bound,
                                      const double size_bound);
}

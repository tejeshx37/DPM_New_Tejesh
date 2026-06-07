#pragma once
#include "cgal-sys/cpp/pair_utils.h"
#include "cgal-sys/cpp/triangulation.h"
#include "cgal-sys/cpp/vector_utils.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace Triangulation {
  using EpickPoint = ::Triangulation::EpickPoint;
  using Face = ::Triangulation::Face;
  using IndexPair = ::Triangulation::IndexPair;
  using Vertex = ::Triangulation::Vertex;
  using Data = ::Triangulation::Data;
  using PointPair = ::Triangulation::PointPair;
  using Constraints = ::Triangulation::Constraints;
}
